"""Feasibility benchmark for the auditable official H6 CEM workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from boundary_jepa.baseline import load_official_pusht_jepa
from boundary_jepa.full_sequence import (
    TracedCEMPlanner,
    denormalize_sequences,
    encode_endpoint_batch,
    encode_goal,
    encode_single_state,
    make_model_scorers,
    sample_actionable_state,
    simulate_sequences,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/jepa_wm_pusht.pth.tar")
    parser.add_argument("--output", default="results/full_sequence/development/benchmark.json")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--sim-candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    started = time.perf_counter()
    bundle = load_official_pusht_jepa(args.checkpoint, device="cuda:0")
    load_seconds = time.perf_counter() - started
    state = sample_actionable_state(args.seed)
    z0 = encode_single_state(bundle, state)
    z_goal = encode_goal(bundle)
    l2, _, _ = make_model_scorers(bundle, z_goal)
    planner = TracedCEMPlanner(
        iterations=args.iterations,
        num_samples=args.samples,
        num_elites=args.elites,
        horizon=6,
        action_dim=bundle.model.action_dim,
        var_scale=1.0,
        device=bundle.model.device,
        seed=args.seed,
    )
    trace = planner.plan(z0, l2)
    sequence_set = np.concatenate([trace.flat_candidates, trace.selected_mean[None]], axis=0)
    subset = sequence_set[: min(args.sim_candidates, len(sequence_set))]
    raw = denormalize_sequences(bundle, subset)
    sim = simulate_sequences(state, raw, seed=args.seed)
    encode_started = time.perf_counter()
    gt_features = encode_endpoint_batch(bundle, sim.final_images, sim.final_states)
    encode_seconds = time.perf_counter() - encode_started
    output = {
        "official_config": {"horizon": 6, "iterations": 30, "samples": 300, "elites": 10},
        "benchmark_config": vars(args),
        "model_action_dim": int(bundle.model.action_dim),
        "raw_control_steps": int(raw.shape[1]),
        "state": state.tolist(),
        "checkpoint": str(bundle.checkpoint),
        "load_seconds": load_seconds,
        "cem_seconds": trace.runtime_seconds,
        "cem_candidates_per_second": args.iterations * args.samples / trace.runtime_seconds,
        "sim_seconds": sim.runtime_seconds,
        "sim_sequences_per_second": len(subset) / sim.runtime_seconds,
        "encode_seconds": encode_seconds,
        "encode_endpoints_per_second": len(subset) / encode_seconds,
        "score_range": [float(trace.scores.min()), float(trace.scores.max())],
        "sim_cost_range": [float(sim.costs.min()), float(sim.costs.max())],
        "gt_feature_shape": list(gt_features.shape),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

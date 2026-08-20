"""Run frozen final fixed-trace attribution and matched-budget CEM repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch
import yaml

from boundary_jepa.baseline import load_official_pusht_jepa
from boundary_jepa.full_sequence import (
    FeatureCapturingScorer,
    TorchReadout,
    TracedCEMPlanner,
    denormalize_sequences,
    encode_endpoint_batch,
    encode_goal,
    encode_single_state,
    sample_actionable_state,
    simulate_sequences,
)


def _planner(cfg: dict, bundle, seed: int) -> TracedCEMPlanner:
    p = cfg["planner"]
    return TracedCEMPlanner(
        iterations=p["iterations"], num_samples=p["samples"], num_elites=p["elites"],
        horizon=cfg["model"]["horizon"], action_dim=bundle.model.action_dim,
        var_scale=p["var_scale"], momentum_mean=p["momentum_mean"],
        momentum_std=p["momentum_std"], device=bundle.model.device, seed=seed,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_sequence/protocol.yaml")
    parser.add_argument("--episode", type=int)
    args = parser.parse_args()
    protocol_path = Path(args.config)
    cfg = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if cfg.get("stage") != "frozen_final_protocol":
        raise RuntimeError("Refusing final evaluation without a frozen final protocol")
    scorer_path = Path(cfg["readout"]["path"])
    if _sha256(scorer_path) != cfg["readout"]["sha256"]:
        raise RuntimeError("Frozen readout hash mismatch")
    development = json.loads(Path(cfg["development_summary"]).read_text(encoding="utf-8"))
    if development["scorer_sha256"] != cfg["readout"]["sha256"]:
        raise RuntimeError("Protocol readout does not match development summary")
    output_dir = Path(cfg["outputs"]["episode_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = [args.episode] if args.episode is not None else range(int(cfg["final_episodes"]))
    bundle = load_official_pusht_jepa(cfg["checkpoint"], device=cfg["device"])
    z_goal = encode_goal(bundle)
    sklearn_readout = joblib.load(scorer_path)
    torch_readout = TorchReadout.from_joblib(scorer_path, bundle.model.device)

    for episode in episodes:
        if episode < 0 or episode >= int(cfg["final_episodes"]):
            raise ValueError(episode)
        path = output_dir / f"episode_{episode:03d}.npz"
        if path.exists():
            print(f"final episode {episode} already exists; skipping", flush=True)
            continue
        episode_seed = int(cfg["final_seed"]) + episode
        state = sample_actionable_state(episode_seed)
        z0 = encode_single_state(bundle, state)

        baseline_scorer = FeatureCapturingScorer(
            bundle, z_goal, alpha=float(cfg["planner"]["terminal_proprio_alpha"]),
            model_query_batch_size=int(cfg["model"]["model_query_batch_size"]),
        )
        baseline_trace = _planner(cfg, bundle, episode_seed).plan(z0, baseline_scorer)
        baseline_actions = np.concatenate(
            [baseline_trace.flat_candidates, baseline_trace.selected_mean[None]], axis=0
        )
        baseline_pred_features = baseline_scorer.stacked_features()
        baseline_l2 = baseline_scorer.stacked_l2_scores()
        baseline_sim = simulate_sequences(
            state, denormalize_sequences(bundle, baseline_actions), seed=episode_seed
        )
        gt_encode_started = time.perf_counter()
        baseline_gt_features = encode_endpoint_batch(
            bundle, baseline_sim.final_images, baseline_sim.final_states,
            batch_size=int(cfg["evaluation"]["gt_encoder_batch_size"]),
        )
        gt_encode_seconds = time.perf_counter() - gt_encode_started
        gt_readout_scores = sklearn_readout.predict(baseline_gt_features).astype(np.float32)
        pred_readout_scores = sklearn_readout.predict(baseline_pred_features).astype(np.float32)
        torch_scores = torch_readout(
            torch.from_numpy(baseline_pred_features).to(bundle.model.device)
        ).cpu().numpy()
        max_readout_error = float(np.max(np.abs(torch_scores - pred_readout_scores)))
        if max_readout_error > float(cfg["evaluation"]["readout_equivalence_tolerance"]):
            raise RuntimeError(f"Torch/sklearn readout mismatch: {max_readout_error}")

        targeted_scorer = FeatureCapturingScorer(
            bundle, z_goal, readout=torch_readout,
            alpha=float(cfg["planner"]["terminal_proprio_alpha"]),
            model_query_batch_size=int(cfg["model"]["model_query_batch_size"]),
        )
        targeted_trace = _planner(cfg, bundle, episode_seed).plan(z0, targeted_scorer)
        targeted_actions = np.concatenate(
            [targeted_trace.flat_candidates, targeted_trace.selected_mean[None]], axis=0
        )
        targeted_sim = simulate_sequences(
            state, denormalize_sequences(bundle, targeted_actions), seed=episode_seed,
            collect_final_images=False,
        )

        base_cost = baseline_sim.costs.astype(np.float64)
        target_cost = targeted_sim.costs.astype(np.float64)
        fixed_min, fixed_max = float(base_cost.min()), float(base_cost.max())
        fixed_denom = max(fixed_max - fixed_min, 1e-8)
        gt_index = int(np.argmin(gt_readout_scores))
        pred_index = int(np.argmin(pred_readout_scores))
        l2_index = int(np.argmin(baseline_l2))
        selected_index = len(base_cost) - 1
        target_selected_index = len(target_cost) - 1
        fixed_regrets = {
            "gt_readout": float((base_cost[gt_index] - fixed_min) / fixed_denom),
            "pred_readout": float((base_cost[pred_index] - fixed_min) / fixed_denom),
            "pred_l2": float((base_cost[l2_index] - fixed_min) / fixed_denom),
        }
        union_min = min(float(base_cost.min()), float(target_cost.min()))
        union_max = max(float(base_cost.max()), float(target_cost.max()))
        union_denom = max(union_max - union_min, 1e-8)
        end_to_end = {
            "baseline_regret": float((base_cost[selected_index] - union_min) / union_denom),
            "targeted_regret": float((target_cost[target_selected_index] - union_min) / union_denom),
            "repair_improvement": float(
                (base_cost[selected_index] - target_cost[target_selected_index]) / union_denom
            ),
            "baseline_raw_cost": float(base_cost[selected_index]),
            "targeted_raw_cost": float(target_cost[target_selected_index]),
            "union_oracle_cost": union_min,
            "union_worst_cost": union_max,
        }
        metrics = {
            "episode": episode,
            "seed": episode_seed,
            "fixed_trace": {
                "regrets": fixed_regrets,
                "prediction_gap": fixed_regrets["pred_readout"] - fixed_regrets["gt_readout"],
                "metric_gap": fixed_regrets["pred_l2"] - fixed_regrets["pred_readout"],
                "indices": {"gt_readout": gt_index, "pred_readout": pred_index, "pred_l2": l2_index},
            },
            "search": {
                "baseline_oracle_cost": float(base_cost.min()),
                "targeted_oracle_cost": float(target_cost.min()),
                "baseline_selection_gap_raw": float(base_cost[selected_index] - base_cost.min()),
                "targeted_selection_gap_raw": float(target_cost[target_selected_index] - target_cost.min()),
                "baseline_coverage_gap_to_union_raw": float(base_cost.min() - union_min),
                "targeted_coverage_gap_to_union_raw": float(target_cost.min() - union_min),
            },
            "end_to_end": end_to_end,
            "timing": {
                "baseline_cem_seconds": baseline_trace.runtime_seconds,
                "baseline_sim_seconds": baseline_sim.runtime_seconds,
                "gt_encode_seconds": gt_encode_seconds,
                "targeted_cem_seconds": targeted_trace.runtime_seconds,
                "targeted_sim_seconds": targeted_sim.runtime_seconds,
            },
            "readout_equivalence_max_abs": max_readout_error,
        }
        np.savez_compressed(
            path,
            state=state,
            baseline_normalized_actions=baseline_actions.astype(np.float32),
            baseline_l2_scores=baseline_l2.astype(np.float32),
            baseline_pred_features=baseline_pred_features.astype(np.float32),
            baseline_pred_readout_scores=pred_readout_scores,
            baseline_gt_readout_scores=gt_readout_scores,
            baseline_simulator_cost=baseline_sim.costs,
            baseline_final_states=baseline_sim.final_states,
            baseline_contact_any=baseline_sim.contact_any,
            baseline_gt_features=baseline_gt_features.astype(np.float32),
            baseline_trace_scores=baseline_trace.scores,
            baseline_means_before=baseline_trace.means_before,
            baseline_stds_before=baseline_trace.stds_before,
            baseline_means_after=baseline_trace.means_after,
            baseline_stds_after=baseline_trace.stds_after,
            targeted_normalized_actions=targeted_actions.astype(np.float32),
            targeted_pred_features=targeted_scorer.stacked_features().astype(np.float32),
            targeted_aligned_scores=targeted_scorer.stacked_returned_scores().astype(np.float32),
            targeted_l2_scores=targeted_scorer.stacked_l2_scores().astype(np.float32),
            targeted_simulator_cost=targeted_sim.costs,
            targeted_final_states=targeted_sim.final_states,
            targeted_contact_any=targeted_sim.contact_any,
            targeted_trace_scores=targeted_trace.scores,
            targeted_means_before=targeted_trace.means_before,
            targeted_stds_before=targeted_trace.stds_before,
            targeted_means_after=targeted_trace.means_after,
            targeted_stds_after=targeted_trace.stds_after,
        )
        (output_dir / f"episode_{episode:03d}.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()

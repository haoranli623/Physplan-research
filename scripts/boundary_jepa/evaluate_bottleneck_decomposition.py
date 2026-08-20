"""Run the frozen Push-T oracle decomposition before any repair experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import joblib
import numpy as np
import pandas as pd
import yaml

from boundary_jepa.analysis import action_set_sha256
from boundary_jepa.decomposition import classify_bottleneck, discrete_cem_search, search_gaps
from boundary_jepa.ranking import bootstrap_mean, endpoint_features, per_state_metrics
from boundary_jepa.sweeps import SPLIT_SEED_OFFSETS, crossing_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bottleneck_decomposition/protocol.yaml")
    return parser.parse_args()


def compact(values: np.ndarray, samples: int, seed: int) -> dict[str, float | int]:
    summary = bootstrap_mean(values, samples=samples, seed=seed)
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    summary.update({
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "q25": float(np.quantile(finite, 0.25)) if finite.size else float("nan"),
        "q75": float(np.quantile(finite, 0.75)) if finite.size else float("nan"),
    })
    return summary


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    result_dir = Path(cfg["results"]["decomposition_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    test = cfg["test"]
    expected_seed = int(cfg["seed"] + SPLIT_SEED_OFFSETS[test["split"]])
    with h5py.File(test["sweeps"], "r") as sweeps, h5py.File(test["features"], "r") as features:
        if int(sweeps.attrs["seed"]) != expected_seed or sweeps.attrs["split"] != test["split"]:
            raise RuntimeError("fresh-test split/seed does not match frozen protocol")
        if int(sweeps.attrs["completed_anchors"]) != int(test["anchors"]):
            raise RuntimeError("test sweep cache is incomplete")
        if int(features.attrs["completed_anchors"]) != int(test["anchors"]):
            raise RuntimeError("test feature cache is incomplete")
        if int(features.attrs.get("autocast_bfloat16", -1)) != 1:
            raise RuntimeError("canonical decomposition requires the frozen BF16 inference path")
        costs = sweeps["task_cost"][:].astype(np.float64)
        modes = sweeps["any_contact"][:].astype(np.int8)
        crossing = sweeps["true_crossing_index"][:].astype(np.int64)
        actions = sweeps["actions"][:]
        offsets = sweeps["offsets"][:]
        anchor_states = sweeps["anchor_states"][:]
        x_gt = endpoint_features(features["z_true_visual_pool"][:], features["z_true_proprio"][:])
        x_pred = endpoint_features(features["z_pred_visual_pool"][:], features["z_pred_proprio"][:])
        l2_gt = features["true_latent_goal_cost"][:].astype(np.float64)
        l2_pred = features["predicted_goal_cost"][:].astype(np.float64)
        feature_runtime = {key: features.attrs.get(key, None) for key in [
            "feature_cache_seconds", "candidate_rollouts_per_second", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"
        ]}
    if np.any(np.ptp(costs, axis=1) < float(cfg["candidate_set"]["informative_cost_range"]) - 1e-7):
        raise RuntimeError("test contains an uninformative state")
    if any(len(crossing_indices(row)) != 1 for row in modes):
        raise RuntimeError("test violates the single-boundary acceptance rule")
    if not np.array_equal(crossing, np.asarray([crossing_indices(row)[0] for row in modes])):
        raise RuntimeError("stored crossing indices disagree with physical labels")
    with h5py.File(cfg["development"]["sweeps"], "r") as development_sweeps:
        dev_states = development_sweeps["anchor_states"][:]
    if any(np.any(np.all(np.isclose(state, dev_states, atol=0, rtol=0), axis=1)) for state in anchor_states):
        raise RuntimeError("exact development/test anchor overlap")
    scorer_path = Path(cfg["development"]["scorer"])
    digest = hashlib.sha256(scorer_path.read_bytes()).hexdigest().upper()
    if digest != cfg["development"]["scorer_sha256"]:
        raise RuntimeError("frozen scorer hash mismatch")
    scorer = joblib.load(scorer_path)
    if not np.isclose(float(scorer.named_steps["ridge"].alpha), float(cfg["scorer"]["alpha"])):
        raise RuntimeError("frozen scorer alpha mismatch")
    states, candidates = costs.shape
    score_gt = scorer.predict(x_gt.reshape(-1, x_gt.shape[-1])).reshape(states, candidates)
    score_pred = scorer.predict(x_pred.reshape(-1, x_pred.shape[-1])).reshape(states, candidates)
    metric_cfg = cfg["metrics"]
    common = {
        "top_k": int(metric_cfg["top_k"]),
        "tie_tolerance": float(metric_cfg["pair_tie_tolerance"]),
        "boundary_near_cells": int(metric_cfg["boundary_near_cells"]),
        "boundary_far_cells": int(metric_cfg["boundary_far_cells"]),
    }
    sources = {"readout_gt": score_gt, "readout_pred": score_pred, "l2_gt": l2_gt, "l2_pred": l2_pred}
    metrics = {name: per_state_metrics(value, costs, modes, crossing, **common) for name, value in sources.items()}
    search_cfg = cfg["search"]
    cem_kwargs = {
        "iterations": int(search_cfg["iterations"]),
        "samples": int(search_cfg["samples_per_iteration"]),
        "elites": int(search_cfg["elites"]),
        "initial_mean": float(search_cfg["initial_mean"]),
        "initial_std": float(search_cfg["initial_std"]),
        "min_std": float(search_cfg["min_std"]),
    }
    searches = [
        discrete_cem_search(
            l2_pred[state], seed=int(cfg["seed"]) + int(search_cfg["seed_offset"]) + state, **cem_kwargs
        ) for state in range(states)
    ]
    search = search_gaps(costs, searches)
    g_pred = metrics["readout_pred"]["selection_regret"] - metrics["readout_gt"]["selection_regret"]
    g_metric = metrics["l2_pred"]["selection_regret"] - metrics["readout_pred"]["selection_regret"]
    samples = int(metric_cfg["bootstrap_samples"])
    seed = int(cfg["seed"])
    metric_summary = {
        source: {
            key: compact(values[key], samples, seed + 100 * i + j)
            for j, key in enumerate(["rho", "pairwise_accuracy", "topk_retrieval", "selection_regret"])
        } for i, (source, values) in enumerate(metrics.items())
    }
    headroom = {
        "representation_readout": {
            "rho_ci_low": metric_summary["readout_gt"]["rho"]["ci_low"],
            "regret_ci_high": metric_summary["readout_gt"]["selection_regret"]["ci_high"],
        },
        "prediction_gap": compact(g_pred, samples, seed + 1000),
        "metric_gap": compact(g_metric, samples, seed + 1001),
        "search_gap": compact(search["g_search"], samples, seed + 1002),
        "selection_gap": compact(search["g_select"], samples, seed + 1003),
    }
    rule = cfg["dominance_rule"]
    diagnosis = classify_bottleneck(
        headroom,
        representation_rho_ci_low=float(rule["representation_min_rho_ci_low"]),
        representation_regret_ci_high=float(rule["representation_max_regret_ci_high"]),
        min_material_gap=float(rule["min_material_gap"]),
        dominance_ratio=float(rule["dominance_ratio"]),
    )
    frame = pd.DataFrame({
        "anchor_state": np.arange(states),
        "cost_range": np.ptp(costs, axis=1),
        "true_crossing_index": crossing,
        "action_set_sha256": [action_set_sha256(value) for value in actions],
        "g_pred": g_pred,
        "g_metric": g_metric,
    })
    for source, values in metrics.items():
        for key, value in values.items():
            frame[f"{source}_{key}"] = value
    for key, value in search.items():
        frame[f"baseline_search_{key}"] = value
    frame.to_csv(result_dir / "state_metrics.csv", index=False)
    np.savez_compressed(
        result_dir / "raw_decomposition.npz", costs=costs, actions=actions, offsets=offsets,
        anchor_states=anchor_states, modes=modes, crossing=crossing, score_gt=score_gt,
        score_pred=score_pred, l2_gt=l2_gt, l2_pred=l2_pred,
        baseline_query_indices=np.stack([item.queried_indices for item in searches]),
        baseline_query_iterations=np.stack([item.queried_iterations for item in searches]),
        baseline_query_scores=np.stack([item.queried_scores for item in searches]),
    )
    summary = {
        "diagnosed_bottleneck": diagnosis,
        "fresh_test_evaluated_once": True,
        "states": states,
        "candidates_per_state": candidates,
        "test_seed": expected_seed,
        "feature_runtime": {key: (value.item() if hasattr(value, "item") else value) for key, value in feature_runtime.items()},
        "metrics": metric_summary,
        "headroom": headroom,
        "baseline_search_total_regret": compact(search["total_search_regret"], samples, seed + 1004),
        "non_additivity_warning": "Controlled headroom estimates interact and are not an additive causal decomposition.",
        "protocol": cfg,
    }
    (result_dir / "decomposition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    repair_map = {
        "DECISION METRIC": (
            "frozen task-aligned readout inside the exact same finite-grid CEM",
            "full-precision predictor inference with the original latent-L2 metric inside the same CEM",
        ),
        "PREDICTION": ("minimal predictor-targeted intervention", "metric replacement"),
        "SEARCH / PROPOSAL": ("matched-budget candidate-coverage intervention", "predictor precision"),
    }
    best, low = repair_map.get(diagnosis, ("no repair until attribution is clear", "no valid low-value repair"))
    prediction = {
        "task": "Push-T local H6 angular reference set",
        "diagnosed_bottleneck": diagnosis,
        "evidence": headroom,
        "predicted_best_repair": best,
        "predicted_low_value_repair": low,
        "repair_results_observed": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": head,
    }
    Path("bottleneck_predictions.json").write_text(json.dumps(prediction, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

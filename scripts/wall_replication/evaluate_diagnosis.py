"""Evaluate the frozen Wall decomposition on D_wall_diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from wall_replication.analysis import (
    bootstrap_mean,
    classify_bottleneck,
    state_metrics,
    subset_metrics,
)
from wall_replication.features import goal_delta_features


def compact(metrics: dict[str, np.ndarray], samples: int, seed: int) -> dict:
    keys = ["rho", "pairwise_accuracy", "topk_retrieval", "regret_pixels", "normalized_regret"]
    return {key: bootstrap_mean(metrics[key], samples=samples, seed=seed + i) for i, key in enumerate(keys)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wall_replication/diagnosis.yaml")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        execution = yaml.safe_load(handle)
    with Path(execution["protocol"]).open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    data = np.load(execution["outputs"]["cache"])
    costs = data["simulator_cost"].astype(np.float64)
    x_gt = goal_delta_features(
        data["true_visual"], data["true_proprio"], data["goal_visual"], data["goal_proprio"]
    )
    x_pred = goal_delta_features(
        data["pred_visual"], data["pred_proprio"], data["goal_visual"], data["goal_proprio"]
    )
    scorer = joblib.load(protocol["development"]["readout"])
    score_gt = scorer.predict(x_gt.reshape(-1, x_gt.shape[-1])).reshape(costs.shape)
    score_pred = scorer.predict(x_pred.reshape(-1, x_pred.shape[-1])).reshape(costs.shape)
    sources = {
        "readout_gt": score_gt,
        "readout_pred": score_pred,
        "l2_gt": data["default_true_score"],
        "l2_pred": data["default_pred_score"],
    }
    metrics = {name: state_metrics(score, costs) for name, score in sources.items()}
    scale = np.maximum(np.ptp(costs, axis=1), 1e-8)
    g_pred = (metrics["readout_pred"]["regret_pixels"] - metrics["readout_gt"]["regret_pixels"]) / scale
    g_metric = (metrics["l2_pred"]["regret_pixels"] - metrics["readout_pred"]["regret_pixels"]) / scale
    query_indices = np.asarray(protocol["candidate_set"]["planner_query_indices"], dtype=np.int64)
    baseline_search = subset_metrics(sources["l2_pred"], costs, query_indices)
    samples = int(protocol["metrics"]["bootstrap_samples"])
    headroom = {
        "representation_readout": {
            "rho": bootstrap_mean(metrics["readout_gt"]["rho"], samples=samples, seed=1),
            "normalized_regret": bootstrap_mean(
                metrics["readout_gt"]["normalized_regret"], samples=samples, seed=2
            ),
        },
        "prediction_gap": bootstrap_mean(g_pred, samples=samples, seed=3),
        "metric_gap": bootstrap_mean(g_metric, samples=samples, seed=4),
        "search_gap": bootstrap_mean(
            baseline_search["normalized_search_gap"], samples=samples, seed=5
        ),
        "selection_gap": bootstrap_mean(
            baseline_search["normalized_selection_gap"], samples=samples, seed=6
        ),
    }
    rule = protocol["dominance_rule"]
    diagnosis = classify_bottleneck(
        headroom,
        min_rho_ci_low=float(rule["representation_min_rho_ci_low"]),
        max_gt_normalized_regret_ci_high=float(
            rule["representation_max_normalized_regret_ci_high"]
        ),
        min_material_gap=float(rule["minimum_material_normalized_gap"]),
        dominance_ratio=float(rule["dominance_ratio"]),
    )
    summary = {
        "frozen_protocol": execution["protocol"],
        "dataset": execution["split"]["name"],
        "anchors": int(len(costs)),
        "candidates_per_anchor": int(costs.shape[1]),
        "diagnosis": diagnosis,
        "metrics": {
            name: compact(value, samples, 100 + 10 * i)
            for i, (name, value) in enumerate(metrics.items())
        },
        "headroom": headroom,
        "baseline_query_planner": {
            "query_budget": len(query_indices),
            "normalized_total_regret": bootstrap_mean(
                baseline_search["normalized_total_regret"], samples=samples, seed=20
            ),
            "pixel_total_regret": bootstrap_mean(
                baseline_search["total_regret_pixels"], samples=samples, seed=21
            ),
        },
    }
    output = Path(execution["outputs"]["result"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    frame = pd.DataFrame(
        {
            "anchor": np.arange(len(costs)),
            "seed": data["seeds"],
            "wall_x": data["wall_x"],
            "door_y": data["door_y"],
            "cost_range_pixels": np.ptp(costs, axis=1),
            "gt_readout_rho": metrics["readout_gt"]["rho"],
            "gt_readout_normalized_regret": metrics["readout_gt"]["normalized_regret"],
            "pred_readout_rho": metrics["readout_pred"]["rho"],
            "pred_readout_normalized_regret": metrics["readout_pred"]["normalized_regret"],
            "pred_l2_rho": metrics["l2_pred"]["rho"],
            "pred_l2_normalized_regret": metrics["l2_pred"]["normalized_regret"],
            "g_pred": g_pred,
            "g_metric": g_metric,
            "g_search": baseline_search["normalized_search_gap"],
            "g_select": baseline_search["normalized_selection_gap"],
        }
    )
    frame.to_csv(output.parent / "state_metrics.csv", index=False)
    np.savez_compressed(
        output.parent / "scores.npz",
        costs=costs,
        score_gt=score_gt,
        score_pred=score_pred,
        l2_gt=sources["l2_gt"],
        l2_pred=sources["l2_pred"],
        query_indices=query_indices,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

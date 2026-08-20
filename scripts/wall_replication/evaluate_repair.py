"""Evaluate the pre-registered predictor repair on unseen D_wall_repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from wall_replication.analysis import bootstrap_mean, state_metrics, subset_metrics
from wall_replication.features import goal_delta_features


def effect_distribution(values: np.ndarray, tolerance: float) -> dict[str, float | int]:
    values = np.asarray(values)
    return {
        "helped_count": int(np.sum(values > tolerance)),
        "unchanged_count": int(np.sum(np.abs(values) <= tolerance)),
        "harmed_count": int(np.sum(values < -tolerance)),
        "helped_fraction": float(np.mean(values > tolerance)),
        "unchanged_fraction": float(np.mean(np.abs(values) <= tolerance)),
        "harmed_fraction": float(np.mean(values < -tolerance)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wall_replication/repair.yaml")
    parser.add_argument(
        "--repaired-predictions", default="artifacts/wall_replication/repair_repaired_predictions.npz"
    )
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        execution = yaml.safe_load(handle)
    with Path(execution["protocol"]).open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    baseline = np.load(execution["outputs"]["cache"])
    repaired = np.load(args.repaired_predictions)
    costs = baseline["simulator_cost"].astype(np.float64)
    x_pred = goal_delta_features(
        baseline["pred_visual"],
        baseline["pred_proprio"],
        baseline["goal_visual"],
        baseline["goal_proprio"],
    )
    scorer = joblib.load(protocol["development"]["readout"])
    wrong_metric_score = scorer.predict(x_pred.reshape(-1, x_pred.shape[-1])).reshape(costs.shape)
    scores = {
        "baseline_l2": baseline["default_pred_score"],
        "targeted_predictor_l2": repaired["default_pred_score"],
        "wrong_metric_readout": wrong_metric_score,
    }
    query_indices = np.asarray(protocol["candidate_set"]["planner_query_indices"], dtype=np.int64)
    full_metrics = {name: state_metrics(score, costs) for name, score in scores.items()}
    query_metrics = {name: subset_metrics(score, costs, query_indices) for name, score in scores.items()}
    baseline_regret = query_metrics["baseline_l2"]["normalized_total_regret"]
    targeted_improvement = baseline_regret - query_metrics["targeted_predictor_l2"][
        "normalized_total_regret"
    ]
    wrong_improvement = baseline_regret - query_metrics["wrong_metric_readout"][
        "normalized_total_regret"
    ]
    targeted_minus_wrong = targeted_improvement - wrong_improvement
    baseline_state_mse = baseline["latent_mse"].mean(axis=1)
    targeted_state_mse = repaired["latent_mse"].mean(axis=1)
    samples = int(protocol["metrics"]["bootstrap_samples"])
    tolerance = float(protocol["repair_success_rule"]["unchanged_tolerance_normalized_regret"])
    method_summary = {}
    for index, name in enumerate(scores):
        method_summary[name] = {
            "full_rho": bootstrap_mean(full_metrics[name]["rho"], samples=samples, seed=100 + index),
            "full_normalized_regret": bootstrap_mean(
                full_metrics[name]["normalized_regret"], samples=samples, seed=110 + index
            ),
            "query_normalized_total_regret": bootstrap_mean(
                query_metrics[name]["normalized_total_regret"], samples=samples, seed=120 + index
            ),
            "query_pixel_total_regret": bootstrap_mean(
                query_metrics[name]["total_regret_pixels"], samples=samples, seed=130 + index
            ),
        }
    targeted_effect = bootstrap_mean(targeted_improvement, samples=samples, seed=200)
    wrong_effect = bootstrap_mean(wrong_improvement, samples=samples, seed=201)
    contrast = bootstrap_mean(targeted_minus_wrong, samples=samples, seed=202)
    rule = protocol["repair_success_rule"]
    checks = {
        "targeted_material": targeted_effect["mean"]
        >= float(rule["targeted_min_mean_normalized_improvement"]),
        "targeted_ci_positive": targeted_effect["ci_low"] > 0.0,
        "targeted_exceeds_wrong_layer": contrast["mean"]
        >= float(rule["targeted_minus_wrong_layer_min_mean"]),
    }
    result = {
        "dataset": execution["split"]["name"],
        "anchors": int(len(costs)),
        "out_of_sample": True,
        "prediction_commit": "8559594b94846b33e84292b60dfdb308bb927b18",
        "pre_registered_diagnosis": "PREDICTION",
        "methods": method_summary,
        "targeted_improvement": {
            **targeted_effect,
            **effect_distribution(targeted_improvement, tolerance),
        },
        "wrong_layer_improvement": {
            **wrong_effect,
            **effect_distribution(wrong_improvement, tolerance),
        },
        "targeted_minus_wrong_layer": contrast,
        "secondary_prediction_error": {
            "baseline_full_trajectory_visual_mse": bootstrap_mean(
                baseline_state_mse, samples=samples, seed=203
            ),
            "targeted_full_trajectory_visual_mse": bootstrap_mean(
                targeted_state_mse, samples=samples, seed=204
            ),
            "mse_reduction": bootstrap_mean(
                baseline_state_mse - targeted_state_mse, samples=samples, seed=205
            ),
            "note": "Secondary diagnostic; not part of the pre-registered repair success rule."
        },
        "success_checks": checks,
        "pre_registered_repair_success": bool(all(checks.values())),
    }
    output = Path(execution["outputs"]["result"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    frame = pd.DataFrame(
        {
            "anchor": np.arange(len(costs)),
            "seed": baseline["seeds"],
            "wall_x": baseline["wall_x"],
            "door_y": baseline["door_y"],
            "cost_range_pixels": np.ptp(costs, axis=1),
            "baseline_normalized_regret": query_metrics["baseline_l2"]["normalized_total_regret"],
            "targeted_normalized_regret": query_metrics["targeted_predictor_l2"][
                "normalized_total_regret"
            ],
            "wrong_metric_normalized_regret": query_metrics["wrong_metric_readout"][
                "normalized_total_regret"
            ],
            "targeted_improvement": targeted_improvement,
            "wrong_metric_improvement": wrong_improvement,
            "targeted_minus_wrong": targeted_minus_wrong,
            "baseline_latent_mse": baseline_state_mse,
            "targeted_latent_mse": targeted_state_mse,
        }
    )
    frame.to_csv(output.parent / "state_metrics.csv", index=False)
    np.savez_compressed(
        output.parent / "scores.npz",
        costs=costs,
        query_indices=query_indices,
        baseline_l2=scores["baseline_l2"],
        targeted_predictor_l2=scores["targeted_predictor_l2"],
        wrong_metric_readout=scores["wrong_metric_readout"],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

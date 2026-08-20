"""Evaluate prospective targeted and wrong-layer repairs after diagnosis commit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import yaml

from boundary_jepa.decomposition import discrete_cem_search, search_gaps
from boundary_jepa.ranking import bootstrap_mean, endpoint_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bottleneck_decomposition/protocol.yaml")
    parser.add_argument("--stage", choices=["targeted", "final"], required=True)
    return parser.parse_args()


def compact(values: np.ndarray, samples: int, seed: int) -> dict[str, float | int]:
    summary = bootstrap_mean(values, samples=samples, seed=seed)
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    summary["median"] = float(np.median(finite)) if finite.size else float("nan")
    return summary


def run_search(score: np.ndarray, cfg: dict) -> tuple[list, dict[str, np.ndarray]]:
    search_cfg = cfg["search"]
    kwargs = {
        "iterations": int(search_cfg["iterations"]),
        "samples": int(search_cfg["samples_per_iteration"]),
        "elites": int(search_cfg["elites"]),
        "initial_mean": float(search_cfg["initial_mean"]),
        "initial_std": float(search_cfg["initial_std"]),
        "min_std": float(search_cfg["min_std"]),
    }
    results = [
        discrete_cem_search(
            score[state], seed=int(cfg["seed"]) + int(search_cfg["seed_offset"]) + state, **kwargs
        ) for state in range(score.shape[0])
    ]
    return results, search_gaps(np.load("results/bottleneck_decomposition/test/raw_decomposition.npz")["costs"], results)


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    prediction = json.loads(Path("bottleneck_predictions.json").read_text(encoding="utf-8"))
    if prediction["diagnosed_bottleneck"] != "DECISION METRIC" or prediction["repair_results_observed"]:
        raise RuntimeError("the committed prospective decision-metric diagnosis is required")
    raw = np.load("results/bottleneck_decomposition/test/raw_decomposition.npz")
    costs = raw["costs"]
    baseline_results, baseline = run_search(raw["l2_pred"], cfg)
    if not np.array_equal(
        np.stack([item.queried_indices for item in baseline_results]), raw["baseline_query_indices"]
    ):
        raise RuntimeError("baseline search does not reproduce the frozen diagnosis")
    targeted_results, targeted = run_search(raw["score_pred"], cfg)
    out = Path(cfg["results"]["repair_dir"])
    out.mkdir(parents=True, exist_ok=True)
    samples = int(cfg["metrics"]["bootstrap_samples"])
    seed = int(cfg["seed"])
    targeted_improvement = baseline["total_search_regret"] - targeted["total_search_regret"]
    targeted_summary = {
        "stage": "targeted_repair_observed_after_diagnosis_commit",
        "diagnosis_commit": "de4d8ca1a2c74ebbf860e80ce4dc6ff76e44d44d",
        "baseline_search_regret": compact(baseline["total_search_regret"], samples, seed + 2000),
        "targeted_readout_search_regret": compact(targeted["total_search_regret"], samples, seed + 2001),
        "targeted_improvement": compact(targeted_improvement, samples, seed + 2002),
        "matched_query_budget": int(cfg["search"]["total_queries_including_final"]),
        "matched_search_seeds": True,
    }
    repair_cfg = cfg["repair_protocol"]
    targeted_summary["passes_frozen_targeted_gate"] = bool(
        targeted_summary["targeted_improvement"]["mean"] >= float(repair_cfg["targeted_min_mean_improvement"])
        and targeted_summary["targeted_improvement"]["ci_low"] > 0.0
    )
    (out / "targeted_summary.json").write_text(json.dumps(targeted_summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / "targeted_raw.npz",
        baseline_query_indices=np.stack([item.queried_indices for item in baseline_results]),
        targeted_query_indices=np.stack([item.queried_indices for item in targeted_results]),
        baseline_selected_index=baseline["selected_index"],
        targeted_selected_index=targeted["selected_index"],
        baseline_search_regret=baseline["total_search_regret"],
        targeted_search_regret=targeted["total_search_regret"],
        targeted_improvement=targeted_improvement,
    )
    frame = pd.DataFrame({
        "anchor_state": np.arange(costs.shape[0]),
        "baseline_regret": baseline["total_search_regret"],
        "targeted_regret": targeted["total_search_regret"],
        "targeted_improvement": targeted_improvement,
        "baseline_selected_index": baseline["selected_index"],
        "targeted_selected_index": targeted["selected_index"],
        "reference_oracle_index": baseline["reference_oracle_index"],
    })
    if args.stage == "targeted":
        frame.to_csv(out / "targeted_state_metrics.csv", index=False)
        print(json.dumps(targeted_summary, indent=2))
        return

    wrong_path = Path(cfg["test"]["wrong_layer_features"])
    if not wrong_path.is_file():
        raise FileNotFoundError(wrong_path)
    with h5py.File(wrong_path, "r") as wrong_features:
        if int(wrong_features.attrs.get("autocast_bfloat16", -1)) != 0:
            raise RuntimeError("wrong-layer cache must use full precision inference")
        l2_fp32 = wrong_features["predicted_goal_cost"][:].astype(np.float64)
        x_fp32 = endpoint_features(
            wrong_features["z_pred_visual_pool"][:], wrong_features["z_pred_proprio"][:]
        )
        wrong_runtime = {key: wrong_features.attrs.get(key, None) for key in [
            "feature_cache_seconds", "candidate_rollouts_per_second", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"
        ]}
    wrong_results, wrong = run_search(l2_fp32, cfg)
    scorer = joblib.load(cfg["development"]["scorer"])
    fp32_readout = scorer.predict(x_fp32.reshape(-1, x_fp32.shape[-1])).reshape(costs.shape)
    predictor_score_delta = np.mean(np.abs(fp32_readout - raw["score_pred"]), axis=1)
    wrong_improvement = baseline["total_search_regret"] - wrong["total_search_regret"]
    wrong_summary = compact(wrong_improvement, samples, seed + 2003)
    target_mean = float(targeted_summary["targeted_improvement"]["mean"])
    wrong_mean = float(wrong_summary["mean"])
    ratio = float(target_mean / max(wrong_mean, 1e-12)) if wrong_mean > 0 else float("inf")
    actionable = bool(
        targeted_summary["passes_frozen_targeted_gate"]
        and target_mean >= float(repair_cfg["actionable_requires_targeted_over_wrong_ratio"]) * max(wrong_mean, 0.0)
    )
    final = {
        **targeted_summary,
        "wrong_layer_fp32_search_regret": compact(wrong["total_search_regret"], samples, seed + 2004),
        "wrong_layer_improvement": wrong_summary,
        "mean_absolute_readout_score_change_bf16_to_fp32": compact(predictor_score_delta, samples, seed + 2005),
        "targeted_over_wrong_mean_improvement_ratio": ratio,
        "wrong_layer_material": bool(wrong_mean >= float(repair_cfg["wrong_layer_material_threshold"])),
        "diagnosis_actionable": actionable,
        "wrong_layer_runtime": {
            key: (value.item() if hasattr(value, "item") else value) for key, value in wrong_runtime.items()
        },
    }
    if actionable:
        final["project_decision"] = "STRONG GO"
    elif targeted_summary["passes_frozen_targeted_gate"]:
        final["project_decision"] = "GO"
    else:
        final["project_decision"] = "INCONCLUSIVE"
    (out / "repair_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    frame["wrong_layer_regret"] = wrong["total_search_regret"]
    frame["wrong_layer_improvement"] = wrong_improvement
    frame["wrong_layer_selected_index"] = wrong["selected_index"]
    frame["predictor_score_delta_bf16_to_fp32"] = predictor_score_delta
    frame.to_csv(out / "state_metrics.csv", index=False)
    np.savez_compressed(
        out / "repair_raw.npz",
        wrong_query_indices=np.stack([item.queried_indices for item in wrong_results]),
        wrong_selected_index=wrong["selected_index"],
        wrong_search_regret=wrong["total_search_regret"],
        wrong_improvement=wrong_improvement,
        l2_fp32=l2_fp32,
        readout_fp32=fp32_readout,
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()

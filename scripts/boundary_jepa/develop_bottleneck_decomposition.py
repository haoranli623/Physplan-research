"""Develop the readout, finite-set decomposition, and search rule on fresh dev states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from boundary_jepa.decomposition import classify_bottleneck, discrete_cem_search, search_gaps
from boundary_jepa.ranking import bootstrap_mean, endpoint_features, per_state_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bottleneck_decomposition/development.yaml")
    return parser.parse_args()


def make_readout(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr", tol=1e-5, max_iter=5000))


def metric_kwargs(cfg: dict) -> dict:
    return {
        "top_k": int(cfg["top_k"]),
        "tie_tolerance": float(cfg["pair_tie_tolerance"]),
        "boundary_near_cells": int(cfg["boundary_near_cells"]),
        "boundary_far_cells": int(cfg["boundary_far_cells"]),
    }


def cem_kwargs(cfg: dict) -> dict:
    return {
        "iterations": int(cfg["iterations"]),
        "samples": int(cfg["samples_per_iteration"]),
        "elites": int(cfg["elites"]),
        "initial_mean": float(cfg["initial_mean"]),
        "initial_std": float(cfg["initial_std"]),
        "min_std": float(cfg["min_std"]),
    }


def compact(values: np.ndarray, samples: int, seed: int) -> dict[str, float | int]:
    summary = bootstrap_mean(values, samples=samples, seed=seed)
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    summary["median"] = float(np.median(finite)) if finite.size else float("nan")
    return summary


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    split = cfg["split"]
    with h5py.File(split["sweeps"], "r") as sweeps, h5py.File(split["features"], "r") as features:
        costs = sweeps["task_cost"][:].astype(np.float64)
        modes = sweeps["any_contact"][:].astype(np.int8)
        crossing = sweeps["true_crossing_index"][:].astype(np.int64)
        actions = sweeps["actions"][:]
        x_gt = endpoint_features(features["z_true_visual_pool"][:], features["z_true_proprio"][:])
        x_pred = endpoint_features(features["z_pred_visual_pool"][:], features["z_pred_proprio"][:])
        l2_gt = features["true_latent_goal_cost"][:].astype(np.float64)
        l2_pred = features["predicted_goal_cost"][:].astype(np.float64)
    states, candidates = costs.shape
    folds = KFold(n_splits=int(cfg["scorer"]["grouped_folds"]), shuffle=True, random_state=int(cfg["seed"]))
    rows: list[dict] = []
    oof: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    common = metric_kwargs(cfg["metrics"])
    for alpha in [float(value) for value in cfg["scorer"]["ridge_alphas"]]:
        score_gt = np.full_like(costs, np.nan)
        score_pred = np.full_like(costs, np.nan)
        for train, valid in folds.split(np.arange(states)):
            scorer = make_readout(alpha)
            scorer.fit(x_gt[train].reshape(-1, x_gt.shape[-1]), costs[train].ravel())
            score_gt[valid] = scorer.predict(x_gt[valid].reshape(-1, x_gt.shape[-1])).reshape(len(valid), candidates)
            score_pred[valid] = scorer.predict(x_pred[valid].reshape(-1, x_pred.shape[-1])).reshape(len(valid), candidates)
        gt_metrics = per_state_metrics(score_gt, costs, modes, crossing, **common)
        pred_metrics = per_state_metrics(score_pred, costs, modes, crossing, **common)
        rows.append({
            "alpha": alpha,
            "mean_gt_rho": float(np.nanmean(gt_metrics["rho"])),
            "mean_pred_rho": float(np.nanmean(pred_metrics["rho"])),
            "mean_gt_regret": float(np.nanmean(gt_metrics["selection_regret"])),
            "mean_pred_regret": float(np.nanmean(pred_metrics["selection_regret"])),
        })
        oof[alpha] = (score_gt, score_pred)
    cv = pd.DataFrame(rows)
    alpha = float(cv.loc[cv["mean_gt_rho"].idxmax(), "alpha"])
    score_gt, score_pred = oof[alpha]
    scorer = make_readout(alpha)
    scorer.fit(x_gt.reshape(-1, x_gt.shape[-1]), costs.ravel())
    joblib.dump(scorer, out / "frozen_gt_latent_ridge.joblib")
    cv.to_csv(out / "alpha_grouped_cv.csv", index=False)

    sources = {"readout_gt": score_gt, "readout_pred": score_pred, "l2_gt": l2_gt, "l2_pred": l2_pred}
    metrics = {name: per_state_metrics(value, costs, modes, crossing, **common) for name, value in sources.items()}
    cem_cfg = cem_kwargs(cfg["search"])
    base_search = [
        discrete_cem_search(l2_pred[state], seed=int(cfg["seed"]) + 10000 + state, **cem_cfg)
        for state in range(states)
    ]
    repaired_search = [
        discrete_cem_search(score_pred[state], seed=int(cfg["seed"]) + 10000 + state, **cem_cfg)
        for state in range(states)
    ]
    base_gaps = search_gaps(costs, base_search)
    repaired_gaps = search_gaps(costs, repaired_search)
    g_pred = metrics["readout_pred"]["selection_regret"] - metrics["readout_gt"]["selection_regret"]
    g_metric = metrics["l2_pred"]["selection_regret"] - metrics["readout_pred"]["selection_regret"]
    samples = int(cfg["metrics"]["bootstrap_samples"])
    summaries = {
        "representation_readout": {
            "rho_ci_low": compact(metrics["readout_gt"]["rho"], samples, int(cfg["seed"]) + 1)["ci_low"],
            "regret_ci_high": compact(metrics["readout_gt"]["selection_regret"], samples, int(cfg["seed"]) + 2)["ci_high"],
        },
        "prediction_gap": compact(g_pred, samples, int(cfg["seed"]) + 3),
        "metric_gap": compact(g_metric, samples, int(cfg["seed"]) + 4),
        "search_gap": compact(base_gaps["g_search"], samples, int(cfg["seed"]) + 5),
        "selection_gap": compact(base_gaps["g_select"], samples, int(cfg["seed"]) + 6),
    }
    rule = cfg["dominance_rule"]
    diagnosis = classify_bottleneck(
        summaries,
        representation_rho_ci_low=float(rule["representation_min_rho_ci_low"]),
        representation_regret_ci_high=float(rule["representation_max_regret_ci_high"]),
        min_material_gap=float(rule["min_material_gap"]),
        dominance_ratio=float(rule["dominance_ratio"]),
    )
    state_frame = pd.DataFrame({"anchor_state": np.arange(states), "cost_range": np.ptp(costs, axis=1)})
    for source, values in metrics.items():
        for key, value in values.items():
            state_frame[f"{source}_{key}"] = value
    state_frame["g_pred"] = g_pred
    state_frame["g_metric"] = g_metric
    for key, value in base_gaps.items():
        state_frame[f"baseline_search_{key}"] = value
    for key, value in repaired_gaps.items():
        state_frame[f"readout_search_{key}"] = value
    state_frame.to_csv(out / "development_state_metrics.csv", index=False)
    np.savez_compressed(
        out / "development_raw.npz",
        costs=costs,
        actions=actions,
        score_gt=score_gt,
        score_pred=score_pred,
        l2_gt=l2_gt,
        l2_pred=l2_pred,
        baseline_query_indices=np.stack([item.queried_indices for item in base_search]),
        baseline_query_iterations=np.stack([item.queried_iterations for item in base_search]),
        readout_query_indices=np.stack([item.queried_indices for item in repaired_search]),
        readout_query_iterations=np.stack([item.queried_iterations for item in repaired_search]),
    )
    full = {
        "development_only": True,
        "states": states,
        "candidates_per_state": candidates,
        "selected_alpha": alpha,
        "cv": rows,
        "metrics": {
            source: {
                key: compact(values[key], samples, int(cfg["seed"]) + 100 + i * 10 + j)
                for j, key in enumerate(["rho", "pairwise_accuracy", "topk_retrieval", "selection_regret"])
            }
            for i, (source, values) in enumerate(metrics.items())
        },
        "headroom": summaries,
        "development_diagnosis": diagnosis,
        "baseline_search_total_regret": compact(base_gaps["total_search_regret"], samples, int(cfg["seed"]) + 20),
        "readout_search_total_regret": compact(repaired_gaps["total_search_regret"], samples, int(cfg["seed"]) + 21),
        "targeted_search_improvement": compact(
            base_gaps["total_search_regret"] - repaired_gaps["total_search_regret"], samples, int(cfg["seed"]) + 22
        ),
        "query_budget": int(cfg["search"]["iterations"] * cfg["search"]["samples_per_iteration"] + 1),
    }
    (out / "development_summary.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    print(json.dumps(full, indent=2))


if __name__ == "__main__":
    main()

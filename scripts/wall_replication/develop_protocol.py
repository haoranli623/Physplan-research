"""Choose Wall readout regularization and inspect rules using D_wall_dev only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import yaml

from wall_replication.analysis import bootstrap_mean, state_metrics, subset_metrics
from wall_replication.features import goal_delta_features


def make_readout(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr", tol=1e-5, max_iter=5000))


def compact_metrics(metrics: dict[str, np.ndarray], samples: int, seed: int) -> dict:
    keys = ["rho", "pairwise_accuracy", "topk_retrieval", "regret_pixels", "normalized_regret"]
    return {key: bootstrap_mean(metrics[key], samples=samples, seed=seed + i) for i, key in enumerate(keys)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wall_replication/development.yaml")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    data = np.load(cfg["outputs"]["cache"])
    costs = data["simulator_cost"].astype(np.float64)
    x_gt = goal_delta_features(
        data["true_visual"], data["true_proprio"], data["goal_visual"], data["goal_proprio"]
    )
    x_pred = goal_delta_features(
        data["pred_visual"], data["pred_proprio"], data["goal_visual"], data["goal_proprio"]
    )
    anchors, candidates = costs.shape
    folds = KFold(
        n_splits=int(cfg["readout"]["grouped_folds"]), shuffle=True, random_state=int(cfg["seed"])
    )
    rows = []
    predictions = {}
    for alpha in [float(value) for value in cfg["readout"]["ridge_alphas"]]:
        gt = np.full_like(costs, np.nan)
        pred = np.full_like(costs, np.nan)
        for train, valid in folds.split(np.arange(anchors)):
            model = make_readout(alpha)
            model.fit(x_gt[train].reshape(-1, x_gt.shape[-1]), costs[train].reshape(-1))
            gt[valid] = model.predict(x_gt[valid].reshape(-1, x_gt.shape[-1])).reshape(len(valid), candidates)
            pred[valid] = model.predict(x_pred[valid].reshape(-1, x_pred.shape[-1])).reshape(len(valid), candidates)
        gt_metrics = state_metrics(gt, costs)
        pred_metrics = state_metrics(pred, costs)
        rows.append(
            {
                "alpha": alpha,
                "mean_gt_rho": float(np.nanmean(gt_metrics["rho"])),
                "mean_pred_rho": float(np.nanmean(pred_metrics["rho"])),
                "mean_gt_normalized_regret": float(gt_metrics["normalized_regret"].mean()),
                "mean_pred_normalized_regret": float(pred_metrics["normalized_regret"].mean()),
            }
        )
        predictions[alpha] = (gt, pred)
    chosen_alpha = max(rows, key=lambda row: (row["mean_gt_rho"], -row["alpha"]))["alpha"]
    score_gt, score_pred = predictions[chosen_alpha]
    frozen = make_readout(chosen_alpha).fit(x_gt.reshape(-1, x_gt.shape[-1]), costs.reshape(-1))
    out = Path("results/wall_replication/development")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(frozen, out / "frozen_gt_latent_ridge.joblib")

    sources = {
        "readout_gt": score_gt,
        "readout_pred": score_pred,
        "l2_gt": data["default_true_score"],
        "l2_pred": data["default_pred_score"],
    }
    samples = int(cfg["metrics"]["bootstrap_samples"])
    metrics = {name: state_metrics(score, costs) for name, score in sources.items()}
    scale = np.maximum(np.ptp(costs, axis=1), 1e-8)
    g_pred = metrics["readout_pred"]["regret_pixels"] - metrics["readout_gt"]["regret_pixels"]
    g_metric = metrics["l2_pred"]["regret_pixels"] - metrics["readout_pred"]["regret_pixels"]
    query_indices = np.asarray(cfg["search"]["query_indices"], dtype=np.int64)
    baseline_search = subset_metrics(sources["l2_pred"], costs, query_indices)
    targeted_search = subset_metrics(sources["readout_pred"], costs, query_indices)
    summary = {
        "development_only": True,
        "anchors": anchors,
        "candidates": candidates,
        "layouts": np.stack([data["wall_x"], data["door_y"]], axis=1).tolist(),
        "alpha_cv": rows,
        "selected_alpha": chosen_alpha,
        "metrics": {
            name: compact_metrics(value, samples, int(cfg["seed"]) + 100 * i)
            for i, (name, value) in enumerate(metrics.items())
        },
        "headroom": {
            "representation_readout": {
                "rho": bootstrap_mean(metrics["readout_gt"]["rho"], samples=samples, seed=1),
                "normalized_regret": bootstrap_mean(metrics["readout_gt"]["normalized_regret"], samples=samples, seed=2),
            },
            "prediction_gap": bootstrap_mean(g_pred / scale, samples=samples, seed=3),
            "metric_gap": bootstrap_mean(g_metric / scale, samples=samples, seed=4),
            "search_gap": bootstrap_mean(baseline_search["normalized_search_gap"], samples=samples, seed=5),
            "selection_gap": bootstrap_mean(baseline_search["normalized_selection_gap"], samples=samples, seed=6),
        },
        "matched_query_repair": {
            "query_budget": len(query_indices),
            "baseline": bootstrap_mean(baseline_search["normalized_total_regret"], samples=samples, seed=7),
            "targeted": bootstrap_mean(targeted_search["normalized_total_regret"], samples=samples, seed=8),
            "improvement": bootstrap_mean(
                baseline_search["normalized_total_regret"] - targeted_search["normalized_total_regret"],
                samples=samples,
                seed=9,
            ),
        },
    }
    (out / "development_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "development_scores.npz",
        costs=costs,
        score_gt=score_gt,
        score_pred=score_pred,
        l2_gt=data["default_true_score"],
        l2_pred=data["default_pred_score"],
        query_indices=query_indices,
        g_pred=g_pred,
        g_metric=g_metric,
        baseline_normalized_total_regret=baseline_search["normalized_total_regret"],
        targeted_normalized_total_regret=targeted_search["normalized_total_regret"],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

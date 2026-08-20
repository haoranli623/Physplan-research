"""Evaluate the predeclared Wall feasibility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import yaml

from wall_replication.features import goal_delta_features


def _rho_rows(score: np.ndarray, cost: np.ndarray) -> np.ndarray:
    values = []
    for s, c in zip(score, cost):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            values.append(float(spearmanr(s, c).statistic))
    return np.asarray(values)


def _metrics(score: np.ndarray, cost: np.ndarray) -> dict[str, float | list[float]]:
    rho = _rho_rows(score, cost)
    selected = np.argmin(score, axis=1)
    oracle = np.argmin(cost, axis=1)
    rows = np.arange(len(cost))
    regret = cost[rows, selected] - cost[rows, oracle]
    cost_range = np.ptp(cost, axis=1)
    normalized_regret = regret / np.maximum(cost_range, 1e-8)
    return {
        "mean_rho": float(np.nanmean(rho)),
        "median_rho": float(np.nanmedian(rho)),
        "mean_regret_pixels": float(regret.mean()),
        "median_regret_pixels": float(np.median(regret)),
        "mean_normalized_regret": float(normalized_regret.mean()),
        "top5_retrieval": float(
            np.mean(
                [bool(set(np.argsort(s)[:5]) & set(np.argsort(c)[:5])) for s, c in zip(score, cost)]
            )
        ),
        "per_state_rho": rho.tolist(),
        "per_state_regret_pixels": regret.tolist(),
        "per_state_normalized_regret": normalized_regret.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wall_replication/feasibility.yaml")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cache_path = Path(cfg["outputs"]["cache"])
    data = np.load(cache_path)
    fit_count = int(cfg["split"]["fit_anchors"])
    heldout = slice(fit_count, fit_count + int(cfg["split"]["heldout_anchors"]))
    true_features = goal_delta_features(
        data["true_visual"], data["true_proprio"], data["goal_visual"], data["goal_proprio"]
    )
    x_fit = true_features[:fit_count].reshape(-1, true_features.shape[-1])
    y_fit = data["simulator_cost"][:fit_count].reshape(-1)
    # Select alpha by five anchor-group folds entirely inside feasibility-fit.
    alphas = [float(value) for value in cfg["readout"]["ridge_alphas"]]
    fold_scores: dict[str, list[float]] = {str(a): [] for a in alphas}
    groups = np.arange(fit_count) % int(cfg["readout"]["grouped_folds"])
    for alpha in alphas:
        for fold in range(int(cfg["readout"]["grouped_folds"])):
            train_anchor = np.flatnonzero(groups != fold)
            val_anchor = np.flatnonzero(groups == fold)
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(
                true_features[train_anchor].reshape(-1, true_features.shape[-1]),
                data["simulator_cost"][train_anchor].reshape(-1),
            )
            score = model.predict(
                true_features[val_anchor].reshape(-1, true_features.shape[-1])
            ).reshape(len(val_anchor), -1)
            fold_scores[str(alpha)].append(float(np.nanmean(_rho_rows(score, data["simulator_cost"][val_anchor]))))
    chosen_alpha = max(alphas, key=lambda a: (np.mean(fold_scores[str(a)]), -a))
    model = make_pipeline(StandardScaler(), Ridge(alpha=chosen_alpha)).fit(x_fit, y_fit)
    heldout_score = model.predict(
        true_features[heldout].reshape(-1, true_features.shape[-1])
    ).reshape(int(cfg["split"]["heldout_anchors"]), -1)
    heldout_cost = data["simulator_cost"][heldout]
    heldout_metrics = _metrics(heldout_score, heldout_cost)
    timing = json.loads(cache_path.with_suffix(".timing.json").read_text(encoding="utf-8"))
    requirements = cfg["feasibility_requirements"]
    checks = {
        "candidate_cost_variation": float(np.median(np.ptp(data["simulator_cost"], axis=1)))
        >= float(requirements["minimum_median_candidate_cost_range_pixels"]),
        "gt_latent_ranking": heldout_metrics["mean_rho"]
        >= float(requirements["minimum_heldout_mean_gt_spearman"]),
        "gt_latent_selection": heldout_metrics["mean_normalized_regret"]
        <= float(requirements["maximum_heldout_mean_normalized_regret"]),
        "simulator_throughput": timing["simulator_candidates_per_second"]
        >= float(requirements["minimum_simulator_candidates_per_second"]),
    }
    total_seconds = (
        timing["simulator_seconds"]
        + timing["render_seconds"]
        + timing["encoder_seconds"]
        + timing["predictor_seconds"]
    )
    estimated_120_hours = total_seconds / timing["anchors"] * 120 / 3600
    checks["estimated_wall_clock"] = estimated_120_hours <= float(
        requirements["maximum_estimated_120_anchor_hours"]
    )
    result = {
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "fit_anchors": fit_count,
        "heldout_anchors": int(cfg["split"]["heldout_anchors"]),
        "candidate_count": int(cfg["candidate_set"]["samples"]),
        "median_candidate_cost_range_pixels": float(np.median(np.ptp(data["simulator_cost"], axis=1))),
        "median_unique_costs_1e3": float(
            np.median([len(np.unique(np.round(row, 3))) for row in data["simulator_cost"]])
        ),
        "collision_fraction": float(data["collisions"].any(axis=2).mean()),
        "crossed_wall_fraction": float(data["crossed_wall"].mean()),
        "chosen_alpha": chosen_alpha,
        "alpha_grouped_cv_mean_rho": {key: float(np.mean(value)) for key, value in fold_scores.items()},
        "heldout_gt_latent_readout": heldout_metrics,
        "heldout_default_gt_l2": _metrics(data["default_true_score"][heldout], heldout_cost),
        "heldout_default_pred_l2": _metrics(data["default_pred_score"][heldout], heldout_cost),
        "timing": timing,
        "estimated_120_anchor_hours": estimated_120_hours,
    }
    output = Path(cfg["outputs"]["result"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

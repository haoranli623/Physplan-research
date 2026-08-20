"""Select and fit the small latent ranking readout on development anchors only."""

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

from boundary_jepa.ranking import endpoint_features, per_state_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ranking_diagnostic/development.yaml")
    return parser.parse_args()


def make_readout(alpha: float):
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha, solver="lsqr", tol=1e-5, max_iter=5000),
    )


def summarize(metrics: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, dict[str, float | int]]:
    output = {}
    for key in ["rho", "pairwise_accuracy", "topk_retrieval", "selection_regret"]:
        values = np.asarray(metrics[key], dtype=np.float64)[mask]
        values = values[np.isfinite(values)]
        output[key] = {
            "states": int(len(values)),
            "mean": float(values.mean()) if len(values) else float("nan"),
            "median": float(np.median(values)) if len(values) else float("nan"),
        }
    return output


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_cfg = cfg["metrics"]
    with h5py.File(cfg["development_sweeps"], "r") as sweeps, h5py.File(
        cfg["development_features"], "r"
    ) as features:
        costs = sweeps["task_cost"][:]
        modes = sweeps["any_contact"][:]
        crossing = sweeps["true_crossing_index"][:]
        x_true = endpoint_features(features["z_true_visual_pool"][:], features["z_true_proprio"][:])
        x_pred = endpoint_features(features["z_pred_visual_pool"][:], features["z_pred_proprio"][:])
        fixed_true = features["true_latent_goal_cost"][:]
        fixed_pred = features["predicted_goal_cost"][:]

    states, candidates = costs.shape
    folds = KFold(
        n_splits=int(cfg["scorer"]["grouped_folds"]),
        shuffle=True,
        random_state=int(cfg["seed"]),
    )
    alpha_rows = []
    oof_by_alpha: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for alpha in [float(value) for value in cfg["scorer"]["ridge_alphas"]]:
        score_true = np.full_like(costs, np.nan, dtype=np.float64)
        score_pred = np.full_like(costs, np.nan, dtype=np.float64)
        for train_states, valid_states in folds.split(np.arange(states)):
            readout = make_readout(alpha)
            readout.fit(x_true[train_states].reshape(-1, x_true.shape[-1]), costs[train_states].ravel())
            score_true[valid_states] = readout.predict(
                x_true[valid_states].reshape(-1, x_true.shape[-1])
            ).reshape(len(valid_states), candidates)
            score_pred[valid_states] = readout.predict(
                x_pred[valid_states].reshape(-1, x_pred.shape[-1])
            ).reshape(len(valid_states), candidates)
        gt_metrics = per_state_metrics(
            score_true,
            costs,
            modes,
            crossing,
            top_k=int(metrics_cfg["top_k"]),
            tie_tolerance=float(metrics_cfg["pair_tie_tolerance"]),
            boundary_near_cells=int(metrics_cfg["boundary_near_cells"]),
            boundary_far_cells=int(metrics_cfg["boundary_far_cells"]),
        )
        pred_metrics = per_state_metrics(
            score_pred,
            costs,
            modes,
            crossing,
            top_k=int(metrics_cfg["top_k"]),
            tie_tolerance=float(metrics_cfg["pair_tie_tolerance"]),
            boundary_near_cells=int(metrics_cfg["boundary_near_cells"]),
            boundary_far_cells=int(metrics_cfg["boundary_far_cells"]),
        )
        alpha_rows.append(
            {
                "alpha": alpha,
                "mean_gt_rho": float(np.nanmean(gt_metrics["rho"])),
                "median_gt_rho": float(np.nanmedian(gt_metrics["rho"])),
                "mean_pred_rho": float(np.nanmean(pred_metrics["rho"])),
                "mean_gt_regret": float(np.nanmean(gt_metrics["selection_regret"])),
                "mean_pred_regret": float(np.nanmean(pred_metrics["selection_regret"])),
            }
        )
        oof_by_alpha[alpha] = (score_true, score_pred)

    alpha_frame = pd.DataFrame(alpha_rows).sort_values("alpha")
    selected_alpha = float(alpha_frame.loc[alpha_frame["mean_gt_rho"].idxmax(), "alpha"])
    score_true, score_pred = oof_by_alpha[selected_alpha]
    final_readout = make_readout(selected_alpha)
    final_readout.fit(x_true.reshape(-1, x_true.shape[-1]), costs.ravel())
    joblib.dump(final_readout, out_dir / "frozen_gt_latent_ridge.joblib")
    alpha_frame.to_csv(out_dir / "alpha_grouped_cv.csv", index=False)
    np.savez_compressed(
        out_dir / "development_oof_scores.npz",
        score_gt=score_true,
        score_pred=score_pred,
        fixed_score_gt=fixed_true,
        fixed_score_pred=fixed_pred,
        simulator_cost=costs,
    )

    informative = np.ptp(costs, axis=1) >= float(metrics_cfg["informative_cost_range"])
    sources = {
        "learned_gt": score_true,
        "learned_pred": score_pred,
        "fixed_gt": fixed_true,
        "fixed_pred": fixed_pred,
    }
    state_frame = pd.DataFrame({"anchor_state": np.arange(states), "informative": informative})
    summaries = {}
    for name, score in sources.items():
        values = per_state_metrics(
            score,
            costs,
            modes,
            crossing,
            top_k=int(metrics_cfg["top_k"]),
            tie_tolerance=float(metrics_cfg["pair_tie_tolerance"]),
            boundary_near_cells=int(metrics_cfg["boundary_near_cells"]),
            boundary_far_cells=int(metrics_cfg["boundary_far_cells"]),
        )
        for key, array in values.items():
            state_frame[f"{name}_{key}"] = array
        summaries[name] = {
            "all_states": summarize(values, np.ones(states, dtype=bool)),
            "informative_states": summarize(values, informative),
        }
    state_frame["delta_rho_gt_minus_pred"] = state_frame["learned_gt_rho"] - state_frame["learned_pred_rho"]
    state_frame["delta_regret_pred_minus_gt"] = (
        state_frame["learned_pred_selection_regret"] - state_frame["learned_gt_selection_regret"]
    )
    state_frame.to_csv(out_dir / "development_state_metrics.csv", index=False)

    summary = {
        "development_only": True,
        "fresh_test_seen": False,
        "states": states,
        "candidates_per_state": candidates,
        "informative_states": int(informative.sum()),
        "cost_range_quantiles": {
            str(q): float(np.quantile(np.ptp(costs, axis=1), q)) for q in [0, 0.25, 0.5, 0.75, 1]
        },
        "selected_alpha": selected_alpha,
        "alpha_selection_rule": "maximum mean per-state GT Spearman under five-fold anchor-grouped OOF evaluation",
        "alpha_results": alpha_rows,
        "metrics": summaries,
    }
    (out_dir / "development_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

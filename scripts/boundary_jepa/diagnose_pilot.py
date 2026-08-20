"""Protocol-preserving diagnostics for an inconclusive Boundary-JEPA pilot.

This script does not replace any primary metric.  It decomposes the frozen
pilot into probe, predictor, and planner/representation floors and records all
100 anchor states, including states without a clean probe crossing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ConstantInputWarning, spearmanr

from boundary_jepa.analysis import bootstrap_spearman


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps", default="artifacts/pilot_cache/evaluation_sweeps.h5")
    parser.add_argument("--features", default="artifacts/pilot_cache/evaluation_features.h5")
    parser.add_argument("--state-metrics", default="results/pilot/state_metrics.csv")
    parser.add_argument("--output-dir", default="results/pilot")
    parser.add_argument("--plots-dir", default="plots/pilot")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)) if values.size else float("nan"),
        "median": float(np.median(values)) if values.size else float("nan"),
        "q25": float(np.quantile(values, 0.25)) if values.size else float("nan"),
        "q75": float(np.quantile(values, 0.75)) if values.size else float("nan"),
        "max": float(np.max(values)) if values.size else float("nan"),
    }


def rowwise_spearman(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    values = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for x, y in zip(left, right):
            values.append(float(spearmanr(x, y).statistic))
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    primary = pd.read_csv(args.state_metrics)
    with h5py.File(args.sweeps, "r") as sweeps, h5py.File(args.features, "r") as features:
        frameskip = int(sweeps.attrs["frameskip"])
        latent_horizon = int(sweeps.attrs["latent_horizon"])
        same_anchor_extension = bool(sweeps.attrs.get("same_anchor_action_extension", 0))
        task_cost = sweeps["task_cost"][:]
        latent_mse = features["latent_mse"][:]
        predicted_goal_cost = features["predicted_goal_cost"][:]
        true_latent_goal_cost = features["true_latent_goal_cost"][:]
        crossing_index = sweeps["true_crossing_index"][:].astype(int)

    states, actions = task_cost.shape
    rows = np.arange(states)
    model_index = np.argmin(predicted_goal_cost, axis=1)
    true_latent_index = np.argmin(true_latent_goal_cost, axis=1)
    oracle_index = np.argmin(task_cost, axis=1)
    model_regret = task_cost[rows, model_index] - task_cost[rows, oracle_index]
    true_latent_planner_regret = task_cost[rows, true_latent_index] - task_cost[rows, oracle_index]
    predictor_incremental_cost = task_cost[rows, model_index] - task_cost[rows, true_latent_index]

    near_boundary_latent_error = np.empty(states, dtype=np.float64)
    for state, crossing in enumerate(crossing_index):
        # Include two actions on either side plus the crossing pair. This is a
        # diagnostic view of the locked pointwise MSE, not a new primary metric.
        start = max(0, crossing - 2)
        stop = min(actions, crossing + 4)
        near_boundary_latent_error[state] = float(latent_mse[state, start:stop].mean())

    diagnostics = primary.copy()
    diagnostics["near_boundary_latent_error"] = near_boundary_latent_error
    diagnostics["true_latent_planner_regret"] = true_latent_planner_regret
    diagnostics["predictor_incremental_task_cost"] = predictor_incremental_cost
    diagnostics["predicted_vs_true_goal_rank_rho"] = rowwise_spearman(
        predicted_goal_cost, true_latent_goal_cost
    )
    diagnostics["true_goal_vs_sim_rank_rho"] = rowwise_spearman(true_latent_goal_cost, task_cost)
    diagnostics["predicted_goal_vs_sim_rank_rho"] = rowwise_spearman(predicted_goal_cost, task_cost)
    diagnostics["clean_gt_and_pred_probe_crossing"] = (
        (diagnostics["gt_probe_crossing_count"] == 1)
        & (diagnostics["pred_probe_crossing_count"] == 1)
    )
    diagnostics.to_csv(output_dir / "state_diagnostics.csv", index=False)

    correlations = {}
    for index, metric in enumerate(
        [
            "probe_floor_boundary_error",
            "wm_boundary_error_vs_probe_gt",
            "boundary_error",
            "latent_error",
            "near_boundary_latent_error",
        ]
    ):
        correlations[metric] = bootstrap_spearman(
            diagnostics[metric].to_numpy(),
            model_regret,
            samples=args.bootstrap_samples,
            seed=args.seed + index,
        )

    clean = diagnostics["clean_gt_and_pred_probe_crossing"].to_numpy(dtype=bool)
    summary = {
        "status": "SECONDARY DIAGNOSTIC; primary pilot remains INCONCLUSIVE",
        "states": int(states),
        "protocol": {
            "latent_horizon": latent_horizon,
            "control_steps": frameskip * latent_horizon,
            "same_anchor_action_extension": same_anchor_extension,
            "boundary_metrics_source": str(Path(args.state_metrics)),
        },
        "model_planner_regret": finite_summary(model_regret),
        "true_latent_planner_floor_regret": finite_summary(true_latent_planner_regret),
        "predictor_incremental_task_cost": finite_summary(predictor_incremental_cost),
        "selection_agreement": {
            "model_equals_true_latent": float(np.mean(model_index == true_latent_index)),
            "model_equals_sim_oracle": float(np.mean(model_index == oracle_index)),
            "true_latent_equals_sim_oracle": float(np.mean(true_latent_index == oracle_index)),
        },
        "statewise_candidate_rank_correlations": {
            "predicted_vs_true_latent_goal_cost": finite_summary(
                diagnostics["predicted_vs_true_goal_rank_rho"].to_numpy()
            ),
            "true_latent_goal_cost_vs_sim_cost": finite_summary(
                diagnostics["true_goal_vs_sim_rank_rho"].to_numpy()
            ),
            "predicted_goal_cost_vs_sim_cost": finite_summary(
                diagnostics["predicted_goal_vs_sim_rank_rho"].to_numpy()
            ),
        },
        "boundary_decomposition": {
            "end_to_end": finite_summary(diagnostics["boundary_error"].to_numpy()),
            "gt_probe_floor": finite_summary(diagnostics["probe_floor_boundary_error"].to_numpy()),
            "predictor_vs_gt_probe": finite_summary(
                diagnostics["wm_boundary_error_vs_probe_gt"].to_numpy()
            ),
            "end_to_end_minus_probe_floor": finite_summary(
                (diagnostics["boundary_error"] - diagnostics["probe_floor_boundary_error"]).to_numpy()
            ),
            "gt_probe_crossing_counts": {
                str(int(k)): int(v)
                for k, v in diagnostics["gt_probe_crossing_count"].value_counts().sort_index().items()
            },
            "pred_probe_crossing_counts": {
                str(int(k)): int(v)
                for k, v in diagnostics["pred_probe_crossing_count"].value_counts().sort_index().items()
            },
            "clean_single_crossing_states": int(clean.sum()),
        },
        "metric_vs_model_regret_bootstrap": correlations,
        "interpretation": (
            "The true-future-latent planner floor is close to the predictor planner regret, so the locked "
            "regret is predominantly limited by goal-representation/objective/candidate alignment. This "
            "diagnosis does not change the primary planner or rescue Q3. Multiple GT-probe crossings also "
            "show that action-level probe quality does not always yield a clean directional boundary."
        ),
    }
    (output_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# Boundary-JEPA Secondary Diagnostic

This report is secondary. The locked primary pilot remains **INCONCLUSIVE**.

- Horizon: {latent_horizon} latent steps / {frameskip * latent_horizon} controls.
- Same-anchor extension: {same_anchor_extension}.
- Predicted-latent planner mean regret: {summary['model_planner_regret']['mean']:.4f}.
- True-latent planner-floor mean regret: {summary['true_latent_planner_floor_regret']['mean']:.4f}.
- Mean predictor incremental task cost: {summary['predictor_incremental_task_cost']['mean']:.4f}.
- Model/true-latent selectors equal simulator oracle on
  {summary['selection_agreement']['model_equals_sim_oracle']:.1%} /
  {summary['selection_agreement']['true_latent_equals_sim_oracle']:.1%} of states.
- End-to-end boundary error vs regret: Spearman
  {correlations['boundary_error']['rho']:.3f}, 95% CI
  [{correlations['boundary_error']['ci_low']:.3f}, {correlations['boundary_error']['ci_high']:.3f}].
- Predictor-vs-GT-probe boundary error vs regret: Spearman
  {correlations['wm_boundary_error_vs_probe_gt']['rho']:.3f}, 95% CI
  [{correlations['wm_boundary_error_vs_probe_gt']['ci_low']:.3f},
  {correlations['wm_boundary_error_vs_probe_gt']['ci_high']:.3f}].
- Clean single-crossing maps in both GT/pred probe: {int(clean.sum())}/{states}.

Interpretation: the true-future-latent planner floor is close to predictor-planner
regret, so the locked regret is predominantly limited by goal representation,
objective, and candidate alignment. This does not rescue Q3 and does not authorize
intervention training.
"""
    (output_dir / "diagnostic_report.md").write_text(report, encoding="utf-8")

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    paired = pd.DataFrame(
        {
            "predicted latent planner": model_regret,
            "true latent planner floor": true_latent_planner_regret,
        }
    ).melt(var_name="selector", value_name="simulator regret")
    sns.boxplot(data=paired, x="selector", y="simulator regret", ax=axes[0], showfliers=False)
    sns.stripplot(data=paired, x="selector", y="simulator regret", ax=axes[0], alpha=0.28, size=2.5)
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].set_title("Planner/representation floor")

    sns.scatterplot(
        data=diagnostics,
        x="probe_floor_boundary_error",
        y="boundary_error",
        hue="pred_probe_crossing_count",
        palette="viridis",
        ax=axes[1],
        legend=False,
    )
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axes[1].set_title("Probe floor vs end-to-end boundary")

    corr_frame = pd.DataFrame(
        {
            "metric": list(correlations),
            "rho": [value["rho"] for value in correlations.values()],
            "low": [value["ci_low"] for value in correlations.values()],
            "high": [value["ci_high"] for value in correlations.values()],
        }
    )
    axes[2].errorbar(
        corr_frame["rho"],
        np.arange(len(corr_frame)),
        xerr=np.vstack([corr_frame["rho"] - corr_frame["low"], corr_frame["high"] - corr_frame["rho"]]),
        fmt="o",
        color="#2c7fb8",
        capsize=3,
    )
    axes[2].axvline(0.0, color="black", linewidth=1)
    axes[2].set_yticks(np.arange(len(corr_frame)), corr_frame["metric"])
    axes[2].set_xlabel("Spearman rho with locked regret (95% CI)")
    axes[2].set_title("State-level diagnostic associations")
    fig.tight_layout()
    fig.savefig(plots_dir / "pilot_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

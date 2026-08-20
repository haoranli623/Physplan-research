"""Compute locked pilot metrics, state-level bootstrap, plots, and report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from boundary_jepa.analysis import (
    action_set_sha256,
    bootstrap_correlation_difference,
    bootstrap_spearman,
    interpolated_boundary,
    normalized_boundary_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/boundary_jepa/pilot.yaml")
    parser.add_argument("--sweeps", default="artifacts/pilot_cache/evaluation_sweeps.h5")
    parser.add_argument("--features", default="artifacts/pilot_cache/evaluation_features.h5")
    parser.add_argument("--probe-dir", default="results/pilot")
    parser.add_argument("--plots-dir", default="plots/pilot")
    return parser.parse_args()


def deterministic_example_indices(frame: pd.DataFrame) -> dict[str, int]:
    latent_rank = frame["latent_error"].rank(pct=True).to_numpy()
    boundary_rank = frame["boundary_error"].rank(pct=True).to_numpy()
    scores = {
        "low latent / high boundary": latent_rank + (1.0 - boundary_rank),
        "high latent / low boundary": (1.0 - latent_rank) + boundary_rank,
        "both good": latent_rank + boundary_rank,
        "both bad": (1.0 - latent_rank) + (1.0 - boundary_rank),
    }
    return {name: int(np.argmin(score)) for name, score in scores.items()}


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    seed = int(cfg["seed"])
    bootstrap_samples = int(cfg["statistics"]["bootstrap_samples"])
    confidence = float(cfg["statistics"]["confidence"])
    probe_dir = Path(args.probe_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    predictions = np.load(probe_dir / "probe_predictions.npz")
    gt_probability = predictions["trajectory_gt_probability"]
    pred_probability = predictions["trajectory_pred_probability"]

    with h5py.File(args.sweeps, "r") as sweeps, h5py.File(args.features, "r") as features:
        offsets = sweeps["offsets"][:]
        actions = sweeps["actions"][:]
        true_mode = sweeps["any_contact"][:].astype(np.int8)
        task_cost = sweeps["task_cost"][:]
        final_coverage = sweeps["final_coverage"][:]
        latent_mse_grid = features["latent_mse"][:]
        predicted_goal_cost = features["predicted_goal_cost"][:]
        true_latent_goal_cost = features["true_latent_goal_cost"][:]

    rows = []
    boundary_locations = []
    for anchor in range(true_mode.shape[0]):
        true_boundary, true_count = interpolated_boundary(offsets[anchor], true_mode[anchor].astype(float))
        gt_probe_boundary, gt_count = interpolated_boundary(
            offsets[anchor], gt_probability[anchor], nearest_to=true_boundary
        )
        pred_probe_boundary, pred_count = interpolated_boundary(
            offsets[anchor], pred_probability[anchor], nearest_to=true_boundary
        )
        model_index = int(np.argmin(predicted_goal_cost[anchor]))
        oracle_index = int(np.argmin(task_cost[anchor]))
        true_latent_index = int(np.argmin(true_latent_goal_cost[anchor]))
        regret = float(task_cost[anchor, model_index] - task_cost[anchor, oracle_index])
        rows.append(
            {
                "anchor_state": anchor,
                "latent_error": float(latent_mse_grid[anchor].mean()),
                "latent_error_final": float(latent_mse_grid[anchor, :, -1].mean()),
                "boundary_error": normalized_boundary_error(pred_probe_boundary, true_boundary, offsets[anchor]),
                "probe_floor_boundary_error": normalized_boundary_error(
                    gt_probe_boundary, true_boundary, offsets[anchor]
                ),
                "wm_boundary_error_vs_probe_gt": (
                    normalized_boundary_error(pred_probe_boundary, gt_probe_boundary, offsets[anchor])
                    if np.isfinite(gt_probe_boundary)
                    else np.nan
                ),
                "mode_agreement_pred": float(np.mean((pred_probability[anchor] >= 0.5) == true_mode[anchor])),
                "mode_agreement_gt_probe": float(np.mean((gt_probability[anchor] >= 0.5) == true_mode[anchor])),
                "planning_regret": max(regret, 0.0),
                "model_selected_index": model_index,
                "oracle_index": oracle_index,
                "true_latent_selected_index": true_latent_index,
                "model_selected_cost": float(task_cost[anchor, model_index]),
                "oracle_cost": float(task_cost[anchor, oracle_index]),
                "model_selected_coverage": float(final_coverage[anchor, model_index]),
                "oracle_coverage": float(final_coverage[anchor, oracle_index]),
                "true_boundary": true_boundary,
                "gt_probe_boundary": gt_probe_boundary,
                "pred_probe_boundary": pred_probe_boundary,
                "true_crossing_count": true_count,
                "gt_probe_crossing_count": gt_count,
                "pred_probe_crossing_count": pred_count,
                "action_set_sha256": action_set_sha256(actions[anchor]),
            }
        )
        boundary_locations.append((true_boundary, gt_probe_boundary, pred_probe_boundary))
    frame = pd.DataFrame(rows)
    frame.to_csv(probe_dir / "state_metrics.csv", index=False)
    predicted_rankings = np.argsort(predicted_goal_cost, axis=1)
    np.savez_compressed(
        probe_dir / "planning_records.npz",
        candidate_actions=actions,
        predicted_goal_cost=predicted_goal_cost,
        predicted_rankings=predicted_rankings,
        simulator_task_cost=task_cost,
        simulator_final_coverage=final_coverage,
        model_selected_index=frame["model_selected_index"].to_numpy(),
        oracle_index=frame["oracle_index"].to_numpy(),
    )

    latent_corr = bootstrap_spearman(
        frame["latent_error"], frame["planning_regret"], samples=bootstrap_samples, seed=seed, confidence=confidence
    )
    boundary_corr = bootstrap_spearman(
        frame["boundary_error"],
        frame["planning_regret"],
        samples=bootstrap_samples,
        seed=seed + 1,
        confidence=confidence,
    )
    correlation_difference = bootstrap_correlation_difference(
        frame["latent_error"],
        frame["boundary_error"],
        frame["planning_regret"],
        samples=bootstrap_samples,
        seed=seed + 2,
        confidence=confidence,
    )
    probe_metrics = json.loads((probe_dir / "probe_metrics.json").read_text(encoding="utf-8"))
    trajectory_gt = probe_metrics["trajectory"]["evaluation_gt"]
    current_gt = probe_metrics["current_only"]["evaluation_gt"]

    # Predeclared, protocol-preserving pilot criteria. They are intentionally not optimized on these results.
    q1 = (
        trajectory_gt["balanced_accuracy"] >= 0.70
        and trajectory_gt["balanced_accuracy"] >= current_gt["balanced_accuracy"] + 0.10
    )
    low_latent_high_boundary_fraction = float(
        np.mean(
            (frame["latent_error"] <= frame["latent_error"].median())
            & (frame["boundary_error"] >= frame["boundary_error"].quantile(0.75))
        )
    )
    q2 = (
        frame["boundary_error"].median() >= frame["probe_floor_boundary_error"].median() + 0.05
        and low_latent_high_boundary_fraction >= 0.10
    )
    q3_point = correlation_difference["rho_difference_boundary_minus_latent"] >= 0.10
    q3_strong = q3_point and correlation_difference["ci_low"] > 0.0
    if q1 and q2 and q3_strong:
        decision = "STRONG GO"
    elif q1 and q2 and q3_point:
        decision = "WEAK GO"
    elif not q1 or (not q2 and frame["boundary_error"].median() <= frame["probe_floor_boundary_error"].median()):
        decision = "NO-GO"
    else:
        decision = "INCONCLUSIVE"

    summary = {
        "decision": decision,
        "anchors": int(len(frame)),
        "actions_per_anchor": int(true_mode.shape[1]),
        "probe": probe_metrics,
        "latent_error": frame["latent_error"].describe().to_dict(),
        "boundary_error": frame["boundary_error"].describe().to_dict(),
        "probe_floor_boundary_error": frame["probe_floor_boundary_error"].describe().to_dict(),
        "planning_regret": frame["planning_regret"].describe().to_dict(),
        "latent_error_vs_regret": latent_corr,
        "boundary_error_vs_regret": boundary_corr,
        "correlation_difference": correlation_difference,
        "low_latent_high_boundary_fraction": low_latent_high_boundary_fraction,
        "go_no_go_questions": {
            "Q1_probe_recovers_mode": bool(q1),
            "Q2_low_error_boundary_misalignment_exists": bool(q2),
            "Q3_boundary_more_associated_with_regret_point": bool(q3_point),
            "Q3_boundary_more_associated_with_regret_strong": bool(q3_strong),
        },
    }
    (probe_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.regplot(data=frame, x="latent_error", y="planning_regret", ax=axes[0], scatter_kws={"s": 24})
    axes[0].set_title(
        f"Pointwise latent error vs regret\nSpearman {latent_corr['rho']:.2f} "
        f"[{latent_corr['ci_low']:.2f}, {latent_corr['ci_high']:.2f}]"
    )
    sns.regplot(data=frame, x="boundary_error", y="planning_regret", ax=axes[1], scatter_kws={"s": 24})
    axes[1].set_title(
        f"Boundary error vs regret\nSpearman {boundary_corr['rho']:.2f} "
        f"[{boundary_corr['ci_low']:.2f}, {boundary_corr['ci_high']:.2f}]"
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "error_vs_regret.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    examples = deterministic_example_indices(frame)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for axis, (name, anchor) in zip(axes.flat, examples.items()):
        x = np.abs(offsets[anchor])
        axis.step(x, true_mode[anchor], where="mid", color="black", label="sim mode", linewidth=2)
        axis.plot(x, gt_probability[anchor], color="#2c7fb8", label="probe(true latent)")
        axis.plot(x, pred_probability[anchor], color="#d95f0e", label="probe(pred latent)")
        axis.axhline(0.5, color="grey", linestyle=":", linewidth=1)
        axis.axvline(x[frame.loc[anchor, "model_selected_index"]], color="red", linestyle="--", label="model action")
        axis.axvline(x[frame.loc[anchor, "oracle_index"]], color="green", linestyle="--", label="oracle action")
        axis.set_title(
            f"{name}; state {anchor}\nlatent={frame.loc[anchor, 'latent_error']:.3g}, "
            f"boundary={frame.loc[anchor, 'boundary_error']:.2f}, regret={frame.loc[anchor, 'planning_regret']:.3f}"
        )
        axis.set_xlabel("absolute angular offset (rad)")
        axis.set_ylim(-0.05, 1.05)
    axes[0, 0].set_ylabel("contact / probability")
    axes[1, 0].set_ylabel("contact / probability")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(plots_dir / "representative_boundary_maps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    sns.histplot(frame["latent_error"], ax=axes[0], bins=20)
    sns.histplot(frame["boundary_error"], ax=axes[1], bins=20)
    sns.histplot(frame["planning_regret"], ax=axes[2], bins=20)
    axes[0].set_title("Per-anchor latent error")
    axes[1].set_title("Per-anchor boundary error")
    axes[2].set_title("Per-anchor local regret")
    fig.tight_layout()
    fig.savefig(plots_dir / "metric_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    report = f"""# Boundary-JEPA Push-T Feasibility Pilot

## 1. Research hypothesis

Pointwise latent prediction accuracy can miss decision-critical local counterfactual contact-boundary misalignment.

## 2. Environment and baseline

- Official Meta JEPA-WMs Push-T implementation and official epoch-50 checkpoint.
- Frozen DINOv2 ViT-S/14 target encoder and official six-layer AdaLN action-conditioned predictor.
- Two predicted latent steps; each latent step concatenates five simulator control actions.

## 3. Dataset protocol

- {len(frame)} state-disjoint held-out evaluation anchors.
- {true_mode.shape[1]} cloned actions per one-sided angular sweep.
- Anchors were accepted by a predeclared simulator-only rule requiring exactly one contact/no-contact crossing.
- Probe training/calibration anchors are disjoint from evaluation anchors.

## 4. Physical labels

Primary physical label: simulator-native contact/no-contact from collision-point counts. Persistent/lost-contact were recorded separately but were not used as the primary label. Task coverage/cost remained separate.

## 5. Probe design

The primary probe consumes current latent plus two consecutive latent-transition deltas (visual and proprioceptive), never action. It was trained only on ground-truth latents, calibrated on held-out probe anchors, then frozen.

Held-out GT-latent probe balanced accuracy: **{trajectory_gt['balanced_accuracy']:.3f}**; macro-F1: **{trajectory_gt['macro_f1']:.3f}**; AUROC: **{trajectory_gt['auroc']:.3f}**. Current-state-only balanced accuracy: **{current_gt['balanced_accuracy']:.3f}**.

## 6. Boundary metric

Primary metric: absolute threshold-crossing displacement normalized by the angular sweep span. Missing predicted crossings receive the maximum normalized error of 1. Raw grids and probabilities are retained.

## 7. Planning-regret definition

For the same saved candidate set, the model selects the minimum official latent-L2 goal cost and the oracle selects the minimum simulator cost. Regret is their simulator-cost difference.

## 8. Number of anchor states/actions

{len(frame)} anchors × {true_mode.shape[1]} actions = {len(frame) * true_mode.shape[1]} held-out counterfactual rollouts.

## 9–12. Primary statistics

- Mean latent MSE: {frame['latent_error'].mean():.6f} (median {frame['latent_error'].median():.6f}).
- Mean normalized end-to-end boundary error: {frame['boundary_error'].mean():.3f} (median {frame['boundary_error'].median():.3f}).
- Mean GT-probe boundary floor: {frame['probe_floor_boundary_error'].mean():.3f} (median {frame['probe_floor_boundary_error'].median():.3f}).
- Mean local planning regret: {frame['planning_regret'].mean():.4f} (median {frame['planning_regret'].median():.4f}).

## 13. State-level correlations

- Latent error vs regret: Spearman {latent_corr['rho']:.3f}, {confidence:.0%} bootstrap CI [{latent_corr['ci_low']:.3f}, {latent_corr['ci_high']:.3f}].
- Boundary error vs regret: Spearman {boundary_corr['rho']:.3f}, {confidence:.0%} bootstrap CI [{boundary_corr['ci_low']:.3f}, {boundary_corr['ci_high']:.3f}].
- Difference (boundary minus latent): {correlation_difference['rho_difference_boundary_minus_latent']:.3f}, CI [{correlation_difference['ci_low']:.3f}, {correlation_difference['ci_high']:.3f}].

These are associations across anchor states, not causal claims.

## 14. Representative visualizations

- `plots/pilot/error_vs_regret.png`
- `plots/pilot/representative_boundary_maps.png`
- `plots/pilot/metric_distributions.png`

The four boundary examples are selected deterministically to cover low/high combinations of latent and boundary error, not manually cherry-picked.

## 15. Failure cases

See `results/pilot/state_metrics.csv`, including missing/multiple predicted crossings and all selected/oracle actions. No states were silently removed.

## 16. Decision

**{decision}**

Q1 probe observability: {q1}. Q2 low-error boundary misalignment: {q2}. Q3 stronger boundary/regret association (point/strong): {q3_point}/{q3_strong}.

## 17. Exact recommended next step

If GO/WEAK GO, generate matched random and boundary-targeted counterfactual training caches and compare standard loss against the predeclared relative loss with identical initialization, steps, data count, and planner. If INCONCLUSIVE/NO-GO, inspect the saved probe-floor versus predictor-gap decomposition before any intervention training.
"""
    Path("pilot_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

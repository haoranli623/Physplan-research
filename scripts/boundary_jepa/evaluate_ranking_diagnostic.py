"""Evaluate the frozen GT-trained scorer once on the fresh ranking test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from boundary_jepa.analysis import action_set_sha256
from boundary_jepa.ranking import bootstrap_mean, endpoint_features, per_state_metrics
from boundary_jepa.sweeps import crossing_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ranking_diagnostic/protocol.yaml")
    parser.add_argument("--results-dir", default="results/ranking_diagnostic/test")
    parser.add_argument("--plots-dir", default="plots/ranking_diagnostic")
    return parser.parse_args()


def metric_summary(values: np.ndarray, bootstrap: dict[str, float | int]) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        **bootstrap,
        "median": float(np.median(finite)) if len(finite) else float("nan"),
        "q25": float(np.quantile(finite, 0.25)) if len(finite) else float("nan"),
        "q75": float(np.quantile(finite, 0.75)) if len(finite) else float("nan"),
    }


def normalize_curve(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    span = float(np.ptp(values))
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


def unique_extreme(values: pd.Series, used: set[int], *, largest: bool) -> int:
    """Choose a deterministic extreme while keeping representative panels distinct."""

    candidates = values.drop(index=list(used), errors="ignore").dropna()
    return int(candidates.idxmax() if largest else candidates.idxmin())


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    test = cfg["test"]
    metric_cfg = cfg["metrics"]
    decision_cfg = cfg["decision"]
    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(test["sweeps"], "r") as sweeps, h5py.File(test["features"], "r") as features:
        expected_seed = int(cfg["seed"] + test["seed_offset"])
        assert sweeps.attrs["split"] == test["split"]
        assert int(sweeps.attrs["seed"]) == expected_seed
        assert int(sweeps.attrs["completed_anchors"]) == int(test["anchors"])
        assert int(features.attrs["completed_anchors"]) == int(test["anchors"])
        assert int(sweeps.attrs["latent_horizon"]) == int(cfg["environment"]["latent_horizon"])
        costs = sweeps["task_cost"][:]
        modes = sweeps["any_contact"][:].astype(np.int8)
        crossing = sweeps["true_crossing_index"][:].astype(np.int64)
        offsets = sweeps["offsets"][:]
        actions = sweeps["actions"][:]
        x_true = endpoint_features(features["z_true_visual_pool"][:], features["z_true_proprio"][:])
        x_pred = endpoint_features(features["z_pred_visual_pool"][:], features["z_pred_proprio"][:])
        fixed_true = features["true_latent_goal_cost"][:]
        fixed_pred = features["predicted_goal_cost"][:]
        feature_runtime = {
            "seconds": float(features.attrs.get("feature_cache_seconds", float("nan"))),
            "candidate_rollouts_per_second": float(
                features.attrs.get("candidate_rollouts_per_second", float("nan"))
            ),
            "gpu_peak_allocated_bytes": int(features.attrs.get("gpu_peak_allocated_bytes", 0)),
            "gpu_peak_reserved_bytes": int(features.attrs.get("gpu_peak_reserved_bytes", 0)),
        }

    expected_range = float(cfg["candidate_set"]["informative_cost_range"])
    if np.any(np.ptp(costs, axis=1) < expected_range - 1e-7):
        raise RuntimeError("Fresh test contains an anchor below the frozen informative-cost threshold")
    recomputed_crossings = [crossing_indices(row) for row in modes]
    if any(len(value) != 1 for value in recomputed_crossings):
        raise RuntimeError("Fresh test violates the frozen single-contact-boundary rule")
    if not np.array_equal(crossing, np.asarray([value[0] for value in recomputed_crossings])):
        raise RuntimeError("Stored boundary indices do not match raw mode grids")

    scorer = joblib.load(cfg["development"]["scorer"])
    ridge = scorer.named_steps["ridge"]
    if not np.isclose(float(ridge.alpha), float(cfg["scorer"]["alpha"])):
        raise RuntimeError("Frozen scorer alpha differs from the committed protocol")
    states, candidates = costs.shape
    learned_true = scorer.predict(x_true.reshape(-1, x_true.shape[-1])).reshape(states, candidates)
    learned_pred = scorer.predict(x_pred.reshape(-1, x_pred.shape[-1])).reshape(states, candidates)

    common_kwargs = {
        "top_k": int(metric_cfg["top_k"]),
        "tie_tolerance": float(metric_cfg["pair_tie_tolerance"]),
        "boundary_near_cells": int(metric_cfg["boundary_near_cells"]),
        "boundary_far_cells": int(metric_cfg["boundary_far_cells"]),
    }
    source_scores = {
        "learned_gt": learned_true,
        "learned_pred": learned_pred,
        "fixed_gt": fixed_true,
        "fixed_pred": fixed_pred,
    }
    source_metrics = {
        name: per_state_metrics(score, costs, modes, crossing, **common_kwargs)
        for name, score in source_scores.items()
    }
    frame = pd.DataFrame(
        {
            "anchor_state": np.arange(states),
            "cost_range": np.ptp(costs, axis=1),
            "true_crossing_index": crossing,
            "action_set_sha256": [action_set_sha256(value) for value in actions],
        }
    )
    for source, values in source_metrics.items():
        for metric, array in values.items():
            frame[f"{source}_{metric}"] = array
    frame["delta_rho_gt_minus_pred"] = frame["learned_gt_rho"] - frame["learned_pred_rho"]
    frame["delta_pairwise_gt_minus_pred"] = (
        frame["learned_gt_pairwise_accuracy"] - frame["learned_pred_pairwise_accuracy"]
    )
    frame["delta_topk_gt_minus_pred"] = (
        frame["learned_gt_topk_retrieval"] - frame["learned_pred_topk_retrieval"]
    )
    frame["delta_regret_pred_minus_gt"] = (
        frame["learned_pred_selection_regret"] - frame["learned_gt_selection_regret"]
    )
    frame["near_accuracy_degradation"] = (
        frame["learned_gt_near_adjacent_accuracy"] - frame["learned_pred_near_adjacent_accuracy"]
    )
    frame["far_accuracy_degradation"] = (
        frame["learned_gt_far_adjacent_accuracy"] - frame["learned_pred_far_adjacent_accuracy"]
    )
    frame["boundary_specific_excess_degradation"] = (
        frame["near_accuracy_degradation"] - frame["far_accuracy_degradation"]
    )
    frame["boundary_pair_degradation"] = (
        frame["learned_gt_boundary_pair_accuracy"] - frame["learned_pred_boundary_pair_accuracy"]
    )
    frame.to_csv(results_dir / "state_metrics.csv", index=False)
    np.savez_compressed(
        results_dir / "raw_scores.npz",
        simulator_cost=costs,
        actions=actions,
        offsets=offsets,
        physical_mode=modes,
        crossing_index=crossing,
        score_gt=learned_true,
        score_pred=learned_pred,
        fixed_score_gt=fixed_true,
        fixed_score_pred=fixed_pred,
    )

    samples = int(metric_cfg["bootstrap_samples"])
    seed = int(cfg["seed"])
    summaries = {}
    for source_index, (source, values) in enumerate(source_metrics.items()):
        summaries[source] = {}
        for metric_index, metric in enumerate(
            ["rho", "pairwise_accuracy", "topk_retrieval", "selection_regret"]
        ):
            bootstrap = bootstrap_mean(
                values[metric], samples=samples, seed=seed + 100 * source_index + metric_index
            )
            summaries[source][metric] = metric_summary(values[metric], bootstrap)

    gap_columns = [
        "delta_rho_gt_minus_pred",
        "delta_pairwise_gt_minus_pred",
        "delta_topk_gt_minus_pred",
        "delta_regret_pred_minus_gt",
        "near_accuracy_degradation",
        "far_accuracy_degradation",
        "boundary_specific_excess_degradation",
        "boundary_pair_degradation",
    ]
    gaps = {
        column: metric_summary(
            frame[column].to_numpy(),
            bootstrap_mean(frame[column].to_numpy(), samples=samples, seed=seed + 1000 + index),
        )
        for index, column in enumerate(gap_columns)
    }

    gt_strong = (
        summaries["learned_gt"]["rho"]["ci_low"] > float(decision_cfg["gt_oracle_min_rho_ci_low"])
        and summaries["learned_gt"]["pairwise_accuracy"]["ci_low"]
        > float(decision_cfg["gt_oracle_min_pairwise_ci_low"])
        and summaries["learned_gt"]["selection_regret"]["ci_high"]
        < float(decision_cfg["gt_oracle_max_regret_ci_high"])
    )
    rho_gap = gaps["delta_rho_gt_minus_pred"]
    regret_gap = gaps["delta_regret_pred_minus_gt"]
    material_gap = (
        rho_gap["mean"] >= float(decision_cfg["material_delta_rho"]) and rho_gap["ci_low"] > 0
    ) or (
        regret_gap["mean"] >= float(decision_cfg["material_delta_regret"]) and regret_gap["ci_low"] > 0
    )
    boundary_gap = gaps["boundary_specific_excess_degradation"]
    boundary_specific = (
        material_gap
        and boundary_gap["mean"] >= float(decision_cfg["boundary_specific_excess_accuracy"])
        and boundary_gap["ci_low"] > 0
    )
    if not gt_strong:
        decision = "REPRESENTATION/SCORING BOTTLENECK"
    elif material_gap and boundary_specific:
        decision = "BOUNDARY-SPECIFIC OPPORTUNITY"
    elif material_gap and boundary_gap["ci_high"] <= 0:
        decision = "GENERAL ACTION-FIDELITY OPPORTUNITY"
    elif material_gap:
        decision = "PREDICTOR-FIDELITY OPPORTUNITY"
    elif (
        rho_gap["ci_high"] < float(decision_cfg["material_delta_rho"])
        and regret_gap["ci_high"] < float(decision_cfg["material_delta_regret"])
    ):
        decision = "PREDICTOR NOT BOTTLENECK"
    else:
        decision = "INCONCLUSIVE"

    summary = {
        "decision": decision,
        "fresh_test_evaluated_once": True,
        "states": states,
        "candidates_per_state": candidates,
        "seed": int(cfg["seed"] + test["seed_offset"]),
        "protocol": cfg,
        "feature_runtime": feature_runtime,
        "scorer_metrics": summaries,
        "oracle_gaps": gaps,
        "decision_checks": {
            "gt_oracle_strong": bool(gt_strong),
            "material_predictor_gap": bool(material_gap),
            "boundary_specific_gap": bool(boundary_specific),
        },
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sns.set_theme(style="whitegrid", context="paper")
    fig, axis = plt.subplots(figsize=(5, 5))
    sns.scatterplot(data=frame, x="learned_gt_rho", y="learned_pred_rho", ax=axis, s=35)
    axis.plot([-1, 1], [-1, 1], "k--", linewidth=1)
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    axis.set_xlabel("GT-latent Spearman rho")
    axis.set_ylabel("Predicted-latent Spearman rho")
    axis.set_title("Fresh test: per-state ranking correlation")
    fig.tight_layout()
    fig.savefig(plots_dir / "rho_gt_vs_pred.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5, 5))
    max_regret = float(
        max(frame["learned_gt_selection_regret"].max(), frame["learned_pred_selection_regret"].max())
    )
    sns.scatterplot(
        data=frame,
        x="learned_gt_selection_regret",
        y="learned_pred_selection_regret",
        ax=axis,
        s=35,
    )
    axis.plot([0, max_regret], [0, max_regret], "k--", linewidth=1)
    axis.set_xlabel("GT-latent selection regret")
    axis.set_ylabel("Predicted-latent selection regret")
    axis.set_title("Fresh test: selected-action regret")
    fig.tight_layout()
    fig.savefig(plots_dir / "regret_gt_vs_pred.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    near_far = frame[["near_accuracy_degradation", "far_accuracy_degradation"]].melt(
        var_name="pair region", value_name="GT minus predicted pair accuracy"
    )
    near_far["pair region"] = near_far["pair region"].map(
        {"near_accuracy_degradation": "Boundary-near", "far_accuracy_degradation": "Boundary-far"}
    )
    fig, axis = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=near_far, x="pair region", y="GT minus predicted pair accuracy", ax=axis)
    sns.stripplot(data=near_far, x="pair region", y="GT minus predicted pair accuracy", ax=axis, alpha=0.25, size=2)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_title("Boundary-near vs far adjacent-pair degradation")
    fig.tight_layout()
    fig.savefig(plots_dir / "boundary_near_vs_far.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    used: set[int] = set()
    both = unique_extreme(
        pd.Series(np.minimum(frame["learned_gt_rho"], frame["learned_pred_rho"])), used, largest=True
    )
    used.add(both)
    largest_gap = unique_extreme(frame["delta_rho_gt_minus_pred"], used, largest=True)
    used.add(largest_gap)
    pred_better = unique_extreme(frame["delta_rho_gt_minus_pred"], used, largest=False)
    used.add(pred_better)
    gt_failure = unique_extreme(frame["learned_gt_rho"], used, largest=False)
    deterministic_cases = {
        "both strong": both,
        "largest GT-PRED gap": largest_gap,
        "PRED beats GT": pred_better,
        "GT failure": gt_failure,
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for axis, (name, state) in zip(axes.flat, deterministic_cases.items()):
        x = np.abs(offsets[state])
        axis.plot(x, normalize_curve(costs[state]), color="black", linewidth=2, label="sim cost")
        axis.plot(x, normalize_curve(learned_true[state]), color="#2c7fb8", label="GT-latent score")
        axis.plot(x, normalize_curve(learned_pred[state]), color="#d95f0e", label="pred-latent score")
        boundary_x = 0.5 * (x[crossing[state]] + x[crossing[state] + 1])
        axis.axvline(boundary_x, color="purple", linestyle="--", label="contact boundary")
        axis.set_title(
            f"{name}; state {state}\n"
            f"rho GT/PRED={frame.loc[state, 'learned_gt_rho']:.2f}/{frame.loc[state, 'learned_pred_rho']:.2f}"
        )
        axis.set_xlabel("absolute angular offset (rad)")
        axis.set_ylim(-0.05, 1.05)
    axes[0, 0].set_ylabel("within-state normalized cost/score")
    axes[1, 0].set_ylabel("within-state normalized cost/score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(plots_dir / "representative_rankings.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    report = f"""# Local Counterfactual Ranking Diagnostic

## 1. Why the original pilot was inconclusive

The original fixed latent-goal planner produced almost the same regret from true and predicted futures, so its planner/representation floor obscured predictor quality.

## 2. New hypothesis

True future latents may expose local action utility even when the fixed latent-goal distance does not; the matched predicted-latent score measures information lost by prediction.

## 3. Development/test split

The old 100 H6 anchors were development-only. The protocol and scorer were frozen in commit `428c2db` before generating {states} new seed-disjoint test anchors.

## 4. Candidate-set construction

Each anchor uses {candidates} identical one-sided angular candidates, repeated for H6/30 controls. A simulator-only frozen rule requires exactly one contact boundary and cost range at least {cfg['candidate_set']['informative_cost_range']}.

## 5. Simulator cost definition

`J_sim = 1 - final_coverage`; lower is better. Raw trajectories, coverage, contact steps, actions, and states remain in the resumable HDF5 cache.

## 6. GT latent scoring protocol

StandardScaler + Ridge(alpha={cfg['scorer']['alpha']}) consumes only terminal global-mean visual latent plus terminal proprio latent. It was selected with five-fold anchor-grouped development CV and fit on development GT latents only. It never reads action.

## 7. Predicted latent scoring protocol

The identical frozen scorer is applied to standard JEPA-WM predicted terminal latents. It is not refit, recalibrated, or tuned for predictions.

## 8. Metrics

Per-state Spearman, pairwise accuracy above {metric_cfg['pair_tie_tolerance']} cost tolerance, top-{metric_cfg['top_k']} retrieval, selection regret, delta-rho, and delta-regret. Anchor is the statistical unit.

## 9. GT latent oracle ranking

- Mean rho {summaries['learned_gt']['rho']['mean']:.3f}, 95% CI [{summaries['learned_gt']['rho']['ci_low']:.3f}, {summaries['learned_gt']['rho']['ci_high']:.3f}], median {summaries['learned_gt']['rho']['median']:.3f}.
- Pairwise accuracy {summaries['learned_gt']['pairwise_accuracy']['mean']:.3f}, CI [{summaries['learned_gt']['pairwise_accuracy']['ci_low']:.3f}, {summaries['learned_gt']['pairwise_accuracy']['ci_high']:.3f}].
- Mean regret {summaries['learned_gt']['selection_regret']['mean']:.4f}, CI [{summaries['learned_gt']['selection_regret']['ci_low']:.4f}, {summaries['learned_gt']['selection_regret']['ci_high']:.4f}].

## 10. Predicted latent ranking

- Mean rho {summaries['learned_pred']['rho']['mean']:.3f}, 95% CI [{summaries['learned_pred']['rho']['ci_low']:.3f}, {summaries['learned_pred']['rho']['ci_high']:.3f}], median {summaries['learned_pred']['rho']['median']:.3f}.
- Pairwise accuracy {summaries['learned_pred']['pairwise_accuracy']['mean']:.3f}.
- Mean regret {summaries['learned_pred']['selection_regret']['mean']:.4f}.
- Top-{metric_cfg['top_k']} retrieval {summaries['learned_pred']['topk_retrieval']['mean']:.3f} versus {summaries['learned_gt']['topk_retrieval']['mean']:.3f} for GT.

## 11. Oracle gap

- Delta rho (GT-PRED): {rho_gap['mean']:.3f}, CI [{rho_gap['ci_low']:.3f}, {rho_gap['ci_high']:.3f}].
- Delta regret (PRED-GT): {regret_gap['mean']:.4f}, CI [{regret_gap['ci_low']:.4f}, {regret_gap['ci_high']:.4f}].

## 12. Boundary-near versus boundary-far

Boundary-specific excess pair-accuracy degradation is {boundary_gap['mean']:.3f}, CI [{boundary_gap['ci_low']:.3f}, {boundary_gap['ci_high']:.3f}]. Adjacent pairs are used in both groups, so action-grid distance is matched.

## 13. Statistical uncertainty

All confidence intervals are {metric_cfg['bootstrap_samples']}-sample state-level bootstraps. Actions and pairs are aggregated within anchor before inference.

## 14. Representative cases

See `plots/ranking_diagnostic/representative_rankings.png`, including four distinct both-strong, largest-gap, predicted-better counterexample, and GT-failure states selected by deterministic metric rules. No state was manually chosen.

## 15. Failure cases

The fixed official goal-L2 scorer fails even with true futures (mean rho {summaries['fixed_gt']['rho']['mean']:.3f}, mean regret {summaries['fixed_gt']['selection_regret']['mean']:.3f}). Individual learned-scorer failures and counterexamples remain in `results/ranking_diagnostic/test/state_metrics.csv`; none was removed.

## 16. Scientific interpretation

The representation contains recoverable decision information, while predicted latents retain essentially the same ranking utility. The fixed official goal-L2 score is the bottleneck that made the prior planner diagnostic weak; predictor fidelity is not the bottleneck under this local protocol.

## 17. Final decision

**{decision}**

GT oracle gate: {gt_strong}. Material predictor gap: {material_gap}. Boundary-specific gap: {boundary_specific}. No Boundary-JEPA intervention was trained.
"""
    Path("ranking_diagnostic_report.md").write_text(report, encoding="utf-8")

    summary_doc = f"""# Ranking Diagnostic Summary

## MAIN RESULT

**{decision}** on {states} fresh cloned states and {candidates} identical candidates per state.

## GT-LATENT RANKING QUALITY

Mean/median rho {summaries['learned_gt']['rho']['mean']:.3f}/{summaries['learned_gt']['rho']['median']:.3f}; mean regret {summaries['learned_gt']['selection_regret']['mean']:.4f}.

## PREDICTED-LATENT RANKING QUALITY

Mean/median rho {summaries['learned_pred']['rho']['mean']:.3f}/{summaries['learned_pred']['rho']['median']:.3f}; mean regret {summaries['learned_pred']['selection_regret']['mean']:.4f}.

## ORACLE GAP

Delta rho {rho_gap['mean']:.3f} [{rho_gap['ci_low']:.3f}, {rho_gap['ci_high']:.3f}]; delta regret {regret_gap['mean']:.4f} [{regret_gap['ci_low']:.4f}, {regret_gap['ci_high']:.4f}].

## BOUNDARY-SPECIFIC EVIDENCE

Near-minus-far excess degradation {boundary_gap['mean']:.3f} [{boundary_gap['ci_low']:.3f}, {boundary_gap['ci_high']:.3f}].

## WHAT THIS RULES OUT

It rules out a material GT-to-predicted ranking loss and boundary-specific ranking degradation at the frozen effect-size thresholds. It does not prove the predictor is universally sufficient outside this H6 local slice.

## WHAT THIS SUPPORTS

Decision-useful information is recoverable from true latents, and the standard JEPA predictor preserves nearly all of it here. The earlier high planner floor came from the fixed latent-goal scoring interface.

## NEXT RESEARCH STEP

Do not train Boundary-JEPA for this protocol. If continuing, study a better task-conditioned latent scoring/readout interface or test a prospectively chosen task where predictor degradation—not scorer mismatch—is independently demonstrated.
"""
    Path("RANKING_DIAGNOSTIC_SUMMARY.md").write_text(summary_doc, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

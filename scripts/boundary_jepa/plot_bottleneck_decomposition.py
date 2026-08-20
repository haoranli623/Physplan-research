"""Create the frozen high-information figures and deterministic failure audit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("results/bottleneck_decomposition")
PLOTS = Path("plots/bottleneck_decomposition")


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    span = float(np.ptp(values))
    return (values - values.min()) / span if span else np.zeros_like(values)


def save_system_figure() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    labels = ["Representation\nGT latent", "Prediction\nGT → predicted", "Decision metric\nL2 → task readout", "Search / proposal\nfinite-grid CEM", "Selected action\nsimulator cost"]
    colors = ["#4C78A8", "#72B7B2", "#F2CF5B", "#F58518", "#E45756"]
    xs = np.linspace(0.08, 0.92, len(labels))
    for index, (x, label, color) in enumerate(zip(xs, labels, colors)):
        ax.text(x, 0.62, label, ha="center", va="center", fontsize=10, color="white",
                bbox={"boxstyle": "round,pad=0.55", "fc": color, "ec": "none"})
        if index < len(labels) - 1:
            ax.annotate("", xy=(xs[index + 1] - 0.09, 0.62), xytext=(x + 0.09, 0.62),
                        arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#444444"})
    ax.text(xs[0], 0.18, "same frozen readout on GT", ha="center", fontsize=9)
    ax.text(xs[1], 0.18, "replace GT with prediction", ha="center", fontsize=9)
    ax.text(xs[2], 0.18, "replace L2 with readout", ha="center", fontsize=9)
    ax.text(xs[3], 0.18, "oracle over queried candidates", ha="center", fontsize=9)
    ax.set_title("Controlled oracle substitutions (headroom estimates are not additive)", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS / "system_decomposition.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    decomposition = json.loads((ROOT / "test/decomposition_summary.json").read_text(encoding="utf-8"))
    repair = json.loads((ROOT / "repair/repair_summary.json").read_text(encoding="utf-8"))
    state = pd.read_csv(ROOT / "test/state_metrics.csv")
    repair_state = pd.read_csv(ROOT / "repair/state_metrics.csv")
    raw = np.load(ROOT / "test/raw_decomposition.npz")
    target_raw = np.load(ROOT / "repair/targeted_raw.npz")

    save_system_figure()

    names = ["Prediction", "Decision metric", "Search / proposal", "Selection within search"]
    keys = ["prediction_gap", "metric_gap", "search_gap", "selection_gap"]
    values = np.asarray([decomposition["headroom"][key]["mean"] for key in keys])
    lows = np.asarray([decomposition["headroom"][key]["ci_low"] for key in keys])
    highs = np.asarray([decomposition["headroom"][key]["ci_high"] for key in keys])
    pd.DataFrame({"component": names, "mean": values, "ci_low": lows, "ci_high": highs}).to_csv(
        ROOT / "test/bottleneck_profile.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#72B7B2", "#F2CF5B", "#F58518", "#E45756"]
    ax.bar(names, values, color=colors, edgecolor="white", yerr=np.vstack([values - lows, highs - values]), capsize=5)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_ylabel("Recoverable simulator-cost headroom")
    ax.set_title("Push-T bottleneck profile (100 fresh anchor states)", weight="bold")
    ax.text(0.99, 0.96, "95% state-bootstrap CI\nSelection overlaps with metric effect",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#555555")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(PLOTS / "pusht_bottleneck_profile.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    methods = ["Baseline\nBF16 + L2", "Targeted\nBF16 + readout", "Wrong layer\nFP32 + L2"]
    summaries = [repair["baseline_search_regret"], repair["targeted_readout_search_regret"], repair["wrong_layer_fp32_search_regret"]]
    means = np.asarray([item["mean"] for item in summaries])
    lo = np.asarray([item["ci_low"] for item in summaries])
    hi = np.asarray([item["ci_high"] for item in summaries])
    pd.DataFrame({"method": methods, "mean_regret": means, "ci_low": lo, "ci_high": hi}).to_csv(
        ROOT / "repair/repair_comparison.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(methods, means, color=["#E45756", "#54A24B", "#9D755D"], yerr=np.vstack([means - lo, hi - means]), capsize=5)
    ax.set_ylabel("Mean simulator regret (lower is better)")
    ax.set_title("Prospective repair test: targeted layer matters", weight="bold")
    ax.text(0.98, 0.95, f"targeted improvement = {repair['targeted_improvement']['mean']:.3f}\nwrong-layer improvement = {repair['wrong_layer_improvement']['mean']:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "targeted_repair.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Deterministic representative: a metric-limited state with positive repair,
    # closest to the median improvement in that predeclared subset.
    positive = repair_state[(repair_state["targeted_improvement"] > 0) & (state["g_metric"] >= 0.02)]
    target_median = float(positive["targeted_improvement"].median())
    representative = int((positive["targeted_improvement"] - target_median).abs().idxmin())
    x = np.linspace(0, 1, raw["costs"].shape[1])
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(x, normalize(raw["costs"][representative]), color="#222222", lw=2.2, label="simulator cost")
    ax.plot(x, normalize(raw["score_gt"][representative]), color="#4C78A8", lw=1.7, label="GT readout score")
    ax.plot(x, normalize(raw["score_pred"][representative]), color="#54A24B", lw=1.7, label="predicted readout score")
    ax.plot(x, normalize(raw["l2_pred"][representative]), color="#E45756", lw=1.7, label="predicted latent L2")
    base_queries = np.unique(target_raw["baseline_query_indices"][representative])
    target_queries = np.unique(target_raw["targeted_query_indices"][representative])
    ax.scatter(x[base_queries], np.full(len(base_queries), 1.08), s=18, color="#E45756", marker="|", label="L2-CEM queried")
    ax.scatter(x[target_queries], np.full(len(target_queries), 1.02), s=18, color="#54A24B", marker="|", label="readout-CEM queried")
    for index, color, label in [
        (int(repair_state.loc[representative, "reference_oracle_index"]), "#222222", "reference best"),
        (int(repair_state.loc[representative, "baseline_selected_index"]), "#E45756", "L2 selected"),
        (int(repair_state.loc[representative, "targeted_selected_index"]), "#54A24B", "readout selected"),
    ]:
        ax.axvline(x[index], color=color, ls="--", lw=1.1, alpha=0.85, label=label)
    ax.set_ylim(-0.04, 1.14)
    ax.set_xlabel("Normalized angular action coordinate")
    ax.set_ylabel("Within-state normalized score / cost")
    ax.set_title(f"Representative held-out state {representative}: identical 65-query budget", weight="bold")
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS / "representative_state.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Predeclared categories; all indices are deterministic metric extrema.
    ambiguity = np.abs(state["g_metric"] - state["baseline_search_g_search"])
    failure_cases = {
        "gt_readout_largest_regret": int(state["readout_gt_selection_regret"].idxmax()),
        "largest_prediction_gap": int(state["g_pred"].idxmax()),
        "latent_l2_surprisingly_best": int(state["l2_pred_selection_regret"].idxmin()),
        "search_largest_coverage_gap": int(state["baseline_search_g_search"].idxmax()),
        "most_ambiguous_metric_vs_search": int(ambiguity.idxmin()),
        "targeted_repair_worst_change": int(repair_state["targeted_improvement"].idxmin()),
        "representative_median_positive_repair_with_metric_gap": representative,
    }
    audit = {
        "selection_rules": "deterministic extrema/median rules evaluated over all 100 frozen states",
        "manual_cherry_pick": False,
        "states": failure_cases,
        "targeted_repair_harmed_fraction": float(np.mean(repair_state["targeted_improvement"] < 0)),
        "targeted_repair_unchanged_fraction": float(np.mean(np.isclose(repair_state["targeted_improvement"], 0))),
        "targeted_repair_helped_fraction": float(np.mean(repair_state["targeted_improvement"] > 0)),
    }
    (ROOT / "failure_case_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

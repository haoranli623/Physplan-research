"""Create the four canonical Wall replication figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "baseline": "#6B7280",
    "targeted": "#0072B2",
    "wrong": "#D55E00",
    "oracle": "#009E73",
}


def save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def bottleneck_profile() -> None:
    diagnosis = json.loads(
        Path("results/wall_replication/diagnosis/diagnosis.json").read_text(encoding="utf-8")
    )
    names = ["Prediction", "Decision metric", "Search / proposal"]
    keys = ["prediction_gap", "metric_gap", "search_gap"]
    values = [diagnosis["headroom"][key]["mean"] for key in keys]
    low = [values[i] - diagnosis["headroom"][key]["ci_low"] for i, key in enumerate(keys)]
    high = [diagnosis["headroom"][key]["ci_high"] - values[i] for i, key in enumerate(keys)]
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    ax.bar(names, values, color=[COLORS["targeted"], COLORS["wrong"], "#CC79A7"], width=0.65)
    ax.errorbar(np.arange(3), values, yerr=np.asarray([low, high]), fmt="none", ecolor="black", capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=0.9, label="material threshold")
    ax.set_ylabel("Recoverable normalized regret")
    ax.set_title("Wall diagnosis (80 unseen anchor states)")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, Path("plots/wall_replication/wall_bottleneck_profile.png"))


def repair_validation() -> None:
    result = json.loads(Path("results/wall_replication/repair/repair.json").read_text(encoding="utf-8"))
    state = pd.read_csv("results/wall_replication/repair/state_metrics.csv")
    names = ["Official\npredictor + L2", "Predictor repair\n+ L2", "Wrong-layer\nmetric readout"]
    keys = ["baseline_l2", "targeted_predictor_l2", "wrong_metric_readout"]
    values = [result["methods"][key]["query_normalized_total_regret"]["mean"] for key in keys]
    low = [
        values[i] - result["methods"][key]["query_normalized_total_regret"]["ci_low"]
        for i, key in enumerate(keys)
    ]
    high = [
        result["methods"][key]["query_normalized_total_regret"]["ci_high"] - values[i]
        for i, key in enumerate(keys)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5), gridspec_kw={"width_ratios": [1.0, 1.25]})
    axes[0].bar(
        names, values, color=[COLORS["baseline"], COLORS["targeted"], COLORS["wrong"]], width=0.7
    )
    axes[0].errorbar(np.arange(3), values, yerr=np.asarray([low, high]), fmt="none", ecolor="black", capsize=4)
    axes[0].set_ylabel("Normalized planning regret")
    axes[0].set_title("Unseen repair set")
    axes[0].tick_params(axis="x", labelsize=9)
    rng = np.random.default_rng(0)
    improvements = [state["targeted_improvement"].to_numpy(), state["wrong_metric_improvement"].to_numpy()]
    parts = axes[1].violinplot(improvements, positions=[0, 1], showmeans=False, showmedians=True)
    for body, color in zip(parts["bodies"], [COLORS["targeted"], COLORS["wrong"]]):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    for index, values_i in enumerate(improvements):
        axes[1].scatter(index + rng.normal(0, 0.035, len(values_i)), values_i, s=9, alpha=0.4, color="black")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks([0, 1], ["Predictor repair", "Wrong-layer metric"])
    axes[1].set_ylabel("Improvement over baseline")
    axes[1].set_title("State-level paired effects")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, Path("plots/wall_replication/wall_repair_validation.png"))


def state_figure(anchor: int, output_name: str, title: str) -> None:
    source = np.load("artifacts/wall_replication/repair_features.npz")
    scores = np.load("results/wall_replication/repair/scores.npz")
    query = scores["query_indices"]
    cost = scores["costs"][anchor]
    waypoint = source["waypoint_y"][anchor]
    selected = {
        "Official + L2": int(query[np.argmin(scores["baseline_l2"][anchor, query])]),
        "Predictor repair": int(query[np.argmin(scores["targeted_predictor_l2"][anchor, query])]),
        "Wrong metric": int(query[np.argmin(scores["wrong_metric_readout"][anchor, query])]),
        "Reference oracle": int(np.argmin(cost)),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    axes[0].plot(waypoint, cost, color="black", linewidth=1.6)
    color_map = {
        "Official + L2": COLORS["baseline"],
        "Predictor repair": COLORS["targeted"],
        "Wrong metric": COLORS["wrong"],
        "Reference oracle": COLORS["oracle"],
    }
    for label, index in selected.items():
        axes[0].scatter(waypoint[index], cost[index], s=48, color=color_map[label], label=label, zorder=3)
    door_y = float(source["door_y"][anchor])
    axes[0].axvspan(door_y - 4, door_y + 4, color="#E5E7EB", alpha=0.8, label="door")
    axes[0].set_xlabel("Waypoint y")
    axes[0].set_ylabel("Simulator terminal cost (pixels)")
    axes[0].set_title("Fixed action candidates")
    axes[0].spines[["top", "right"]].set_visible(False)
    wall_x = float(source["wall_x"][anchor])
    axes[1].fill_betweenx([4, door_y - 4], wall_x - 3, wall_x + 3, color="black")
    axes[1].fill_betweenx([door_y + 4, 60], wall_x - 3, wall_x + 3, color="black")
    for label, index in selected.items():
        trajectory = source["sampled_states"][anchor, index]
        axes[1].plot(trajectory[:, 0], trajectory[:, 1], "-o", ms=2.5, lw=1.6, color=color_map[label], label=label)
    start = source["starts"][anchor]
    goal = source["goals"][anchor]
    axes[1].scatter(start[0], start[1], marker="s", s=50, color="#F0E442", edgecolor="black", zorder=5)
    axes[1].scatter(goal[0], goal[1], marker="*", s=100, color=COLORS["oracle"], edgecolor="black", zorder=5)
    axes[1].set_xlim(4, 60)
    axes[1].set_ylim(4, 60)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Selected trajectories")
    axes[1].legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle(title)
    save(fig, Path(f"plots/wall_replication/{output_name}.png"))


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.titlesize": 12})
    bottleneck_profile()
    repair_validation()
    state = pd.read_csv("results/wall_replication/repair/state_metrics.csv")
    helped = state[state["targeted_improvement"] > 1e-6]
    median_value = helped["targeted_improvement"].median()
    representative = int((helped["targeted_improvement"] - median_value).abs().idxmin())
    failure = int(state["targeted_improvement"].idxmin())
    state_figure(representative, "wall_representative_state", f"Representative helped state (anchor {representative})")
    state_figure(failure, "wall_failure_state", f"Counterexample: predictor repair harms (anchor {failure})")
    selection = {"representative_anchor": representative, "failure_anchor": failure}
    Path("plots/wall_replication/selected_states.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

"""Create the single compact full-sequence validation figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/full_sequence/summary.json")
    parser.add_argument("--metrics", default="results/full_sequence/episode_metrics.csv")
    parser.add_argument("--output", default="plots/full_sequence/full_sequence_validation.png")
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    frame = pd.read_csv(args.metrics)
    values = summary["summaries"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15))

    labels = ["Prediction", "Metric"]
    keys = ["prediction_gap", "metric_gap"]
    means = [values[k]["mean"] for k in keys]
    errors = np.array([
        [values[k]["mean"] - values[k]["ci_low"] for k in keys],
        [values[k]["ci_high"] - values[k]["mean"] for k in keys],
    ])
    axes[0].bar(labels, means, color=["#4C78A8", "#F58518"], yerr=errors, capsize=4)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("Normalized regret gap")
    axes[0].set_title("Fixed-trace attribution")

    repair_keys = ["baseline_regret", "targeted_regret"]
    repair_labels = ["Official L2", "Task readout"]
    repair_means = [values[k]["mean"] for k in repair_keys]
    repair_errors = np.array([
        [values[k]["mean"] - values[k]["ci_low"] for k in repair_keys],
        [values[k]["ci_high"] - values[k]["mean"] for k in repair_keys],
    ])
    axes[1].bar(repair_labels, repair_means, color=["#777777", "#54A24B"], yerr=repair_errors, capsize=4)
    axes[1].set_ylabel("Normalized regret")
    axes[1].set_title("Matched-budget adaptive CEM")

    improvements = frame["repair_improvement"].sort_values().to_numpy()
    colors = np.where(improvements > 1e-6, "#54A24B", np.where(improvements < -1e-6, "#E45756", "#999999"))
    axes[2].bar(np.arange(len(improvements)), improvements, color=colors, width=0.9)
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_xlabel("Episode (sorted)")
    axes[2].set_ylabel("Repair improvement")
    axes[2].set_title("State-wise heterogeneity")

    fig.suptitle(f"Official Push-T H6 sequence planning — {summary['decision']}", fontsize=11)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

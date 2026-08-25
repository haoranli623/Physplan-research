"""Rebuild paper figures from frozen JSON summaries; runs no experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def triple(item: dict) -> tuple[float, float, float]:
    return float(item["mean"]), float(item["ci_low"]), float(item["ci_high"])


local = load("results/bottleneck_decomposition/test/decomposition_summary.json")
local_repair = load("results/bottleneck_decomposition/repair/repair_summary.json")
wall = load("results/wall_replication/diagnosis/diagnosis.json")
wall_repair = load("results/wall_replication/repair/repair.json")
h6 = load("results/full_sequence/summary.json")

prediction = [
    triple(local["headroom"]["prediction_gap"]),
    triple(wall["headroom"]["prediction_gap"]),
    triple(h6["summaries"]["prediction_gap"]),
]
metric = [
    triple(local["headroom"]["metric_gap"]),
    triple(wall["headroom"]["metric_gap"]),
    triple(h6["summaries"]["metric_gap"]),
]
repair = [
    triple(local_repair["targeted_improvement"]),
    triple(wall_repair["targeted_improvement"]),
    triple(h6["summaries"]["repair_improvement"]),
]

# Fail loudly if the authoritative artifacts are not the frozen paper inputs.
assert np.allclose([x[0] for x in prediction], [0.0071091199, 0.5635845994, 0.1608075309])
assert np.allclose([x[0] for x in metric], [0.0851215696, -0.0279372807, 0.0949734593])
assert np.allclose([x[0] for x in repair], [0.0815481704, 0.3891065056, 0.0843137159])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

names = ["Local Push-T", "Wall", "Push-T H6"]
blue, orange, green = "#3B6FB6", "#D97904", "#218C74"
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw={"width_ratios": [1.35, 1]})

ax = axes[0]
x = np.arange(len(names))
width = 0.34
for offset, values, color, label in [
    (-width / 2, prediction, blue, "Prediction gap"),
    (width / 2, metric, orange, "Decision-metric gap"),
]:
    means = np.array([v[0] for v in values])
    low = means - np.array([v[1] for v in values])
    high = np.array([v[2] for v in values]) - means
    ax.bar(x + offset, means, width, color=color, label=label, zorder=2)
    ax.errorbar(x + offset, means, yerr=np.vstack([low, high]), fmt="none", ecolor="#222222", capsize=2.5, lw=0.8, zorder=3)
ax.axhline(0, color="#555555", lw=0.7)
ax.set_xticks(x, names)
ax.set_ylabel("Within-regime normalized-regret gap")
ax.set_title("(a) Diagnosed headroom by regime")
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)

ax = axes[1]
means = np.array([v[0] for v in repair])
low = means - np.array([v[1] for v in repair])
high = np.array([v[2] for v in repair]) - means
bars = ax.bar(x, means, 0.58, color=[orange, blue, green], zorder=2)
bars[2].set_hatch("///")
bars[2].set_edgecolor("#333333")
ax.errorbar(x, means, yerr=np.vstack([low, high]), fmt="none", ecolor="#222222", capsize=2.5, lw=0.8, zorder=3)
ax.axhline(0, color="#555555", lw=0.7)
ax.set_xticks(x, names)
ax.set_ylabel("Within-regime regret improvement")
ax.set_title("(b) Frozen interventions")
ax.text(2, means[2] + high[2] + 0.012, "inconclusive", ha="center", va="bottom", fontsize=7)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)

fig.tight_layout(w_pad=1.3)
for suffix in ("pdf", "png"):
    fig.savefig(OUT / f"cross_regime_results.{suffix}", dpi=300, bbox_inches="tight")
plt.close(fig)

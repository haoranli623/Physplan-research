"""Post-hoc state-level audit for the frozen full-sequence result.

This script is descriptive and is not part of the preregistered decision rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, *, samples: int = 5000, seed: int = 20260902
) -> dict[str, float | int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    observed = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(samples):
        index = rng.integers(0, len(x), size=len(x))
        value = float(spearmanr(x[index], y[index]).statistic)
        if np.isfinite(value):
            draws.append(value)
    return {
        "rho": observed,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "bootstrap_samples": samples,
        "finite_draws": len(draws),
        "seed": seed,
    }


def ranked_cases(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    groups = [
        ("most_harmed", frame.sort_values("repair_improvement", ascending=True)),
        ("most_helped", frame.sort_values("repair_improvement", ascending=False)),
        (
            "prediction_dominant",
            frame.assign(score=frame["prediction_gap"] - frame["metric_gap"]).sort_values(
                "score", ascending=False
            ),
        ),
        (
            "metric_dominant",
            frame.assign(score=frame["metric_gap"] - frame["prediction_gap"]).sort_values(
                "score", ascending=False
            ),
        ),
    ]
    rows = []
    for role, ordered in groups:
        for rank, (_, row) in enumerate(ordered.head(count).iterrows(), start=1):
            item = row.to_dict()
            item.update({"audit_role": role, "audit_rank": rank})
            rows.append(item)
    result = pd.DataFrame(rows)
    preferred = ["audit_role", "audit_rank", "episode", "seed"]
    return result[preferred + [column for column in result.columns if column not in preferred]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="results/full_sequence/episode_metrics.csv")
    parser.add_argument(
        "--summary-output", default="results/full_sequence/exploratory_failure_audit.json"
    )
    parser.add_argument("--cases-output", default="results/full_sequence/failure_cases.csv")
    parser.add_argument("--plot-output", default="plots/full_sequence/statewise_regimes.png")
    parser.add_argument("--case-count", type=int, default=5)
    args = parser.parse_args()

    frame = pd.read_csv(args.metrics)
    if len(frame) != 30:
        raise RuntimeError(f"Expected the frozen 30 episodes, found {len(frame)}")

    correlations = {
        "metric_gap_vs_repair": bootstrap_spearman(
            frame["metric_gap"].to_numpy(), frame["repair_improvement"].to_numpy()
        ),
        "prediction_gap_vs_repair": bootstrap_spearman(
            frame["prediction_gap"].to_numpy(),
            frame["repair_improvement"].to_numpy(),
            seed=20260903,
        ),
        "metric_minus_prediction_vs_repair": bootstrap_spearman(
            (frame["metric_gap"] - frame["prediction_gap"]).to_numpy(),
            frame["repair_improvement"].to_numpy(),
            seed=20260904,
        ),
    }
    audit = {
        "status": "POST-HOC EXPLORATORY; NOT PART OF THE FROZEN DECISION RULE",
        "episodes": len(frame),
        "correlations": correlations,
        "interpretation_warning": (
            "Correlations are descriptive, share quantities with the repair evaluation, and do not "
            "establish causality or a deployable state-wise selector."
        ),
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    cases = ranked_cases(frame, args.case_count)
    cases_path = Path(args.cases_output)
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(cases_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.65))
    color = frame["repair_improvement"].to_numpy()
    bound = max(abs(float(color.min())), abs(float(color.max())))
    scatter = axes[0].scatter(
        frame["prediction_gap"], frame["metric_gap"], c=color, cmap="RdYlGn",
        vmin=-bound, vmax=bound, edgecolor="black", linewidth=0.35,
    )
    low = min(float(frame["prediction_gap"].min()), float(frame["metric_gap"].min()))
    high = max(float(frame["prediction_gap"].max()), float(frame["metric_gap"].max()))
    axes[0].plot([low, high], [low, high], "--", color="#666666", linewidth=1)
    axes[0].axhline(0, color="#999999", linewidth=0.7)
    axes[0].axvline(0, color="#999999", linewidth=0.7)
    axes[0].set_xlabel("Prediction gap")
    axes[0].set_ylabel("Metric gap")
    axes[0].set_title("State-wise failure regimes")
    fig.colorbar(scatter, ax=axes[0], label="Repair improvement")

    axes[1].scatter(
        frame["metric_gap"], frame["repair_improvement"], c="#4C78A8",
        edgecolor="black", linewidth=0.35,
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axvline(0, color="#999999", linewidth=0.7)
    rho = correlations["metric_gap_vs_repair"]
    axes[1].set_xlabel("Metric gap")
    axes[1].set_ylabel("Repair improvement")
    axes[1].set_title(
        f"Metric headroom vs repair\nSpearman $\\rho$={rho['rho']:.2f} "
        f"[{rho['ci_low']:.2f}, {rho['ci_high']:.2f}]"
    )
    for episode in [1, 17, 20, 27]:
        row = frame.loc[frame["episode"] == episode].iloc[0]
        axes[1].annotate(
            str(episode), (row["metric_gap"], row["repair_improvement"]),
            xytext=(4, 4), textcoords="offset points", fontsize=8,
        )

    fig.suptitle("Exploratory state-level audit (not preregistered)", fontsize=11)
    fig.tight_layout()
    plot_path = Path(args.plot_output)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

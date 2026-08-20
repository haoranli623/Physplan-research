"""State-level ranking, regret, and bootstrap utilities for Wall."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr


def state_metrics(scores: np.ndarray, costs: np.ndarray, *, top_k: int = 5) -> dict[str, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    if scores.shape != costs.shape or scores.ndim != 2:
        raise ValueError("scores and costs must share [anchor, candidate] shape")
    rho = []
    pairwise = []
    retrieval = []
    selected = np.argmin(scores, axis=1)
    oracle = np.argmin(costs, axis=1)
    for score, cost in zip(scores, costs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            rho.append(float(spearmanr(score, cost).statistic))
        upper = np.triu(np.ones((len(cost), len(cost)), dtype=bool), 1)
        cost_diff = cost[:, None] - cost[None, :]
        score_diff = score[:, None] - score[None, :]
        informative = upper & (np.abs(cost_diff) > 0.05)
        pairwise.append(float(np.mean(np.sign(cost_diff[informative]) == np.sign(score_diff[informative]))))
        retrieval.append(bool(set(np.argsort(score)[:top_k]) & set(np.argsort(cost)[:top_k])))
    rows = np.arange(len(costs))
    regret = costs[rows, selected] - costs[rows, oracle]
    ranges = np.ptp(costs, axis=1)
    return {
        "rho": np.asarray(rho),
        "pairwise_accuracy": np.asarray(pairwise),
        "topk_retrieval": np.asarray(retrieval, dtype=np.float64),
        "selected_index": selected,
        "oracle_index": oracle,
        "regret_pixels": regret,
        "normalized_regret": regret / np.maximum(ranges, 1e-8),
    }


def subset_metrics(scores: np.ndarray, costs: np.ndarray, indices: np.ndarray) -> dict[str, np.ndarray]:
    subset_score = scores[:, indices]
    subset_cost = costs[:, indices]
    metrics = state_metrics(subset_score, subset_cost, top_k=min(5, len(indices)))
    metrics["selected_index"] = indices[metrics["selected_index"]]
    metrics["oracle_index"] = indices[metrics["oracle_index"]]
    full_oracle_cost = costs.min(axis=1)
    subset_oracle_cost = subset_cost.min(axis=1)
    selected_cost = costs[np.arange(len(costs)), metrics["selected_index"]]
    ranges = np.maximum(np.ptp(costs, axis=1), 1e-8)
    metrics["search_gap_pixels"] = subset_oracle_cost - full_oracle_cost
    metrics["normalized_search_gap"] = metrics["search_gap_pixels"] / ranges
    metrics["selection_gap_pixels"] = selected_cost - subset_oracle_cost
    metrics["normalized_selection_gap"] = metrics["selection_gap_pixels"] / ranges
    metrics["total_regret_pixels"] = selected_cost - full_oracle_cost
    metrics["normalized_total_regret"] = metrics["total_regret_pixels"] / ranges
    return metrics


def bootstrap_mean(values: np.ndarray, *, samples: int, seed: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": float("nan"), "median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "states": 0}
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "states": int(len(values)),
    }


def classify_bottleneck(
    summary: dict,
    *,
    min_rho_ci_low: float,
    max_gt_normalized_regret_ci_high: float,
    min_material_gap: float,
    dominance_ratio: float,
) -> str:
    ceiling = summary["representation_readout"]
    if (
        ceiling["rho"]["ci_low"] < min_rho_ci_low
        or ceiling["normalized_regret"]["ci_high"] > max_gt_normalized_regret_ci_high
    ):
        return "REPRESENTATION / READOUT"
    gaps = {
        "PREDICTION": summary["prediction_gap"],
        "DECISION METRIC": summary["metric_gap"],
        "SEARCH / PROPOSAL": summary["search_gap"],
    }
    material = {
        name: value
        for name, value in gaps.items()
        if value["mean"] >= min_material_gap and value["ci_low"] > 0.0
    }
    if material:
        ordered = sorted(material.items(), key=lambda item: item[1]["mean"], reverse=True)
        runner = max([value["mean"] for _, value in ordered[1:]] or [0.0])
        if ordered[0][1]["mean"] >= dominance_ratio * max(runner, 1e-12):
            return ordered[0][0]
    selection = summary["selection_gap"]
    if selection["mean"] >= min_material_gap and selection["ci_low"] > 0 and not material:
        return "SELECTION"
    if not material and selection["mean"] < min_material_gap:
        return "NO CLEAR DOMINANT BOTTLENECK"
    return "MIXED / UNCERTAIN"

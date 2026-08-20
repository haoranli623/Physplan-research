"""Controlled finite-set bottleneck decomposition for local Push-T planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiscreteCEMResult:
    """All observable decisions from one finite-grid CEM search."""

    queried_indices: np.ndarray
    queried_iterations: np.ndarray
    queried_scores: np.ndarray
    selected_index: int
    final_mean_coordinate: float


def discrete_cem_search(
    scores: np.ndarray,
    *,
    iterations: int,
    samples: int,
    elites: int,
    seed: int,
    initial_mean: float = 0.5,
    initial_std: float = 0.5,
    min_std: float = 0.03,
) -> DiscreteCEMResult:
    """Run CEM on a normalized scalar coordinate and snap queries to a fixed grid.

    ``scores`` are visible only when a grid point is queried. The final updated
    mean is appended as one explicit query, so the selected action always belongs
    to the auditable candidate set.
    """

    score = np.asarray(scores, dtype=np.float64)
    if score.ndim != 1 or score.size < 2 or not np.all(np.isfinite(score)):
        raise ValueError("scores must be a finite one-dimensional grid")
    if iterations < 1 or samples < 2 or not 1 <= elites < samples:
        raise ValueError("invalid CEM budget")
    grid = np.linspace(0.0, 1.0, score.size)
    rng = np.random.default_rng(seed)
    mean = float(initial_mean)
    std = float(initial_std)
    query_indices: list[int] = []
    query_iterations: list[int] = []
    for iteration in range(iterations):
        coordinates = np.clip(rng.normal(mean, std, size=samples), 0.0, 1.0)
        coordinates[0] = np.clip(mean, 0.0, 1.0)
        indices = np.abs(coordinates[:, None] - grid[None]).argmin(axis=1)
        visible_scores = score[indices]
        elite_positions = np.argsort(visible_scores, kind="stable")[:elites]
        elite_coordinates = grid[indices[elite_positions]]
        mean = float(np.mean(elite_coordinates))
        std = max(float(np.std(elite_coordinates)), float(min_std))
        query_indices.extend(indices.tolist())
        query_iterations.extend([iteration] * samples)
    selected = int(np.abs(grid - np.clip(mean, 0.0, 1.0)).argmin())
    query_indices.append(selected)
    query_iterations.append(iterations)
    query_array = np.asarray(query_indices, dtype=np.int64)
    return DiscreteCEMResult(
        queried_indices=query_array,
        queried_iterations=np.asarray(query_iterations, dtype=np.int64),
        queried_scores=score[query_array],
        selected_index=selected,
        final_mean_coordinate=mean,
    )


def search_gaps(
    simulator_cost: np.ndarray,
    results: list[DiscreteCEMResult],
) -> dict[str, np.ndarray]:
    """Compute finite-reference proposal and selection gaps per anchor."""

    costs = np.asarray(simulator_cost, dtype=np.float64)
    if costs.ndim != 2 or len(results) != costs.shape[0]:
        raise ValueError("one search result is required per state")
    reference_index = np.argmin(costs, axis=1)
    reference_cost = costs[np.arange(costs.shape[0]), reference_index]
    search_oracle_index = np.empty(costs.shape[0], dtype=np.int64)
    selected_index = np.empty(costs.shape[0], dtype=np.int64)
    for state, result in enumerate(results):
        unique = np.unique(result.queried_indices)
        best_local = int(np.argmin(costs[state, unique]))
        search_oracle_index[state] = int(unique[best_local])
        selected_index[state] = int(result.selected_index)
    search_oracle_cost = costs[np.arange(costs.shape[0]), search_oracle_index]
    selected_cost = costs[np.arange(costs.shape[0]), selected_index]
    return {
        "reference_oracle_index": reference_index,
        "reference_oracle_cost": reference_cost,
        "search_oracle_index": search_oracle_index,
        "search_oracle_cost": search_oracle_cost,
        "selected_index": selected_index,
        "selected_cost": selected_cost,
        "g_search": search_oracle_cost - reference_cost,
        "g_select": selected_cost - search_oracle_cost,
        "total_search_regret": selected_cost - reference_cost,
        "unique_queries": np.asarray(
            [len(np.unique(result.queried_indices)) for result in results], dtype=np.int64
        ),
    }


def classify_bottleneck(
    summaries: dict[str, dict[str, float]],
    *,
    representation_rho_ci_low: float,
    representation_regret_ci_high: float,
    min_material_gap: float,
    dominance_ratio: float,
) -> str:
    """Apply the predeclared effect-size rule to bootstrap summaries."""

    gt = summaries["representation_readout"]
    if gt["rho_ci_low"] <= representation_rho_ci_low or gt["regret_ci_high"] >= representation_regret_ci_high:
        return "REPRESENTATION / READOUT"

    gaps = {
        "PREDICTION": summaries["prediction_gap"],
        "DECISION METRIC": summaries["metric_gap"],
        "SEARCH / PROPOSAL": summaries["search_gap"],
    }
    material = {
        name: value
        for name, value in gaps.items()
        if value["mean"] >= min_material_gap and value["ci_low"] > 0.0
    }
    if material:
        ordered = sorted(material.items(), key=lambda item: item[1]["mean"], reverse=True)
        winner, best = ordered[0]
        runner_mean = max([value["mean"] for _, value in ordered[1:]] or [0.0])
        if best["mean"] >= dominance_ratio * max(runner_mean, 1e-12):
            return winner

    select = summaries["selection_gap"]
    metric = summaries["metric_gap"]
    if (
        select["mean"] >= min_material_gap
        and select["ci_low"] > 0.0
        and metric["mean"] < min_material_gap
    ):
        return "SELECTION"
    if not material and select["mean"] < min_material_gap:
        return "NO CLEAR DOMINANT BOTTLENECK"
    return "MIXED / UNCERTAIN"

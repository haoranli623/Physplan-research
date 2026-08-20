"""Small, action-blind readouts and state-level local ranking metrics."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr


def endpoint_features(visual: np.ndarray, proprio: np.ndarray) -> np.ndarray:
    """Concatenate terminal pooled visual/proprio latents, never raw action."""

    visual = np.asarray(visual, dtype=np.float32)
    proprio = np.asarray(proprio, dtype=np.float32)
    if visual.ndim != 4 or proprio.ndim != 4 or visual.shape[:3] != proprio.shape[:3]:
        raise ValueError(f"Expected matching [state, action, time, dim] arrays: {visual.shape}, {proprio.shape}")
    return np.concatenate([visual[:, :, -1], proprio[:, :, -1]], axis=-1)


def _rank_correlation(score: np.ndarray, cost: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        value = spearmanr(score, cost).statistic
    return float(value)


def _pair_accuracy(score_diff: np.ndarray, cost_diff: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(score_diff[mask]) == np.sign(cost_diff[mask])))


def per_state_metrics(
    scores: np.ndarray,
    costs: np.ndarray,
    modes: np.ndarray,
    crossing_indices: np.ndarray,
    *,
    top_k: int,
    tie_tolerance: float,
    boundary_near_cells: int,
    boundary_far_cells: int,
) -> dict[str, np.ndarray]:
    """Compute metrics per independent anchor state.

    Spearman uses score/cost with the same direction (lower is better). Pair
    statistics are first aggregated within state; callers bootstrap states.
    Boundary-near/far comparisons use adjacent pairs only, holding action-grid
    distance fixed.
    """

    scores = np.asarray(scores, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    modes = np.asarray(modes, dtype=np.int8)
    crossing_indices = np.asarray(crossing_indices, dtype=np.int64)
    if scores.shape != costs.shape or modes.shape != costs.shape:
        raise ValueError("scores, costs, and modes must share [state, action] shape")
    states, actions = costs.shape
    if not 1 <= top_k <= actions:
        raise ValueError("top_k must lie within the candidate count")

    result: dict[str, list[float | int]] = {
        "rho": [],
        "pairwise_accuracy": [],
        "topk_retrieval": [],
        "selected_index": [],
        "oracle_index": [],
        "selection_regret": [],
        "boundary_pair_accuracy": [],
        "near_adjacent_accuracy": [],
        "far_adjacent_accuracy": [],
        "near_pair_count": [],
        "far_pair_count": [],
    }
    for state in range(states):
        score = scores[state]
        cost = costs[state]
        result["rho"].append(_rank_correlation(score, cost))
        score_diff = score[:, None] - score[None, :]
        cost_diff = cost[:, None] - cost[None, :]
        upper = np.triu(np.ones((actions, actions), dtype=bool), k=1)
        informative = upper & (np.abs(cost_diff) > tie_tolerance)
        result["pairwise_accuracy"].append(_pair_accuracy(score_diff, cost_diff, informative))

        selected = int(np.argmin(score))
        oracle = int(np.argmin(cost))
        result["selected_index"].append(selected)
        result["oracle_index"].append(oracle)
        result["selection_regret"].append(float(max(cost[selected] - cost[oracle], 0.0)))
        score_top = set(np.argsort(score)[:top_k].tolist())
        cost_top = set(np.argsort(cost)[:top_k].tolist())
        result["topk_retrieval"].append(float(bool(score_top & cost_top)))

        edge_cost = np.diff(cost)
        edge_score = np.diff(score)
        edge_informative = np.abs(edge_cost) > tie_tolerance
        edge_index = np.arange(actions - 1)
        crossing = int(crossing_indices[state])
        if crossing < 0 or crossing >= actions - 1:
            boundary_mask = np.zeros(actions - 1, dtype=bool)
            near_mask = np.zeros(actions - 1, dtype=bool)
            far_mask = np.zeros(actions - 1, dtype=bool)
        else:
            boundary_mask = (edge_index == crossing) & (modes[state, :-1] != modes[state, 1:])
            near_mask = np.abs(edge_index - crossing) <= boundary_near_cells
            far_mask = (np.abs(edge_index - crossing) >= boundary_far_cells) & (
                modes[state, :-1] == modes[state, 1:]
            )
        boundary_mask &= edge_informative
        near_mask &= edge_informative
        far_mask &= edge_informative
        result["boundary_pair_accuracy"].append(
            _pair_accuracy(edge_score, edge_cost, boundary_mask)
        )
        result["near_adjacent_accuracy"].append(_pair_accuracy(edge_score, edge_cost, near_mask))
        result["far_adjacent_accuracy"].append(_pair_accuracy(edge_score, edge_cost, far_mask))
        result["near_pair_count"].append(int(near_mask.sum()))
        result["far_pair_count"].append(int(far_mask.sum()))

    integer_keys = {"selected_index", "oracle_index", "near_pair_count", "far_pair_count"}
    return {
        key: np.asarray(value, dtype=np.int64 if key in integer_keys else np.float64)
        for key, value in result.items()
    }


def bootstrap_mean(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Bootstrap a state-level mean after removing undefined states."""

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "states": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    boot = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha)),
        "states": int(len(values)),
    }


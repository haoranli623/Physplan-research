"""Auditable boundary, regret, and state-level bootstrap statistics."""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.stats import spearmanr


def interpolated_boundary(
    offsets: np.ndarray,
    values: np.ndarray,
    *,
    threshold: float = 0.5,
    nearest_to: float | None = None,
) -> tuple[float, int]:
    """Interpolate threshold crossings and select the one nearest a reference."""

    offsets = np.asarray(offsets, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    signs = values >= threshold
    crossing_indices = np.flatnonzero(signs[:-1] != signs[1:])
    locations = []
    for index in crossing_indices:
        left_value, right_value = values[index], values[index + 1]
        if right_value == left_value:
            fraction = 0.5
        else:
            fraction = (threshold - left_value) / (right_value - left_value)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        locations.append(offsets[index] + fraction * (offsets[index + 1] - offsets[index]))
    if not locations:
        return float("nan"), 0
    locations_array = np.asarray(locations)
    if nearest_to is None:
        selected = 0
    else:
        selected = int(np.argmin(np.abs(locations_array - nearest_to)))
    return float(locations_array[selected]), len(locations)


def normalized_boundary_error(predicted: float, reference: float, offsets: np.ndarray) -> float:
    span = float(np.ptp(offsets))
    if span <= 0:
        raise ValueError("Sweep span must be positive")
    if not np.isfinite(predicted):
        return 1.0
    return float(min(abs(predicted - reference) / span, 1.0))


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    estimate = float(spearmanr(x, y).statistic) if len(x) > 2 else float("nan")
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(samples):
        indices = rng.integers(0, len(x), len(x))
        value = spearmanr(x[indices], y[indices]).statistic
        if np.isfinite(value):
            boot.append(float(value))
    alpha = (1.0 - confidence) / 2.0
    return {
        "rho": estimate,
        "ci_low": float(np.quantile(boot, alpha)) if boot else float("nan"),
        "ci_high": float(np.quantile(boot, 1.0 - alpha)) if boot else float("nan"),
        "states": int(len(x)),
    }


def bootstrap_correlation_difference(
    latent_error: np.ndarray,
    boundary_error: np.ndarray,
    regret: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    latent_error = np.asarray(latent_error)
    boundary_error = np.asarray(boundary_error)
    regret = np.asarray(regret)
    valid = np.isfinite(latent_error) & np.isfinite(boundary_error) & np.isfinite(regret)
    latent_error, boundary_error, regret = latent_error[valid], boundary_error[valid], regret[valid]
    point = float(spearmanr(boundary_error, regret).statistic - spearmanr(latent_error, regret).statistic)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(samples):
        indices = rng.integers(0, len(regret), len(regret))
        rb = spearmanr(boundary_error[indices], regret[indices]).statistic
        rl = spearmanr(latent_error[indices], regret[indices]).statistic
        if np.isfinite(rb) and np.isfinite(rl):
            boot.append(float(rb - rl))
    alpha = (1.0 - confidence) / 2.0
    return {
        "rho_difference_boundary_minus_latent": point,
        "ci_low": float(np.quantile(boot, alpha)) if boot else float("nan"),
        "ci_high": float(np.quantile(boot, 1.0 - alpha)) if boot else float("nan"),
        "states": int(len(regret)),
    }


def action_set_sha256(actions: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(actions, dtype=np.float32).tobytes()).hexdigest()

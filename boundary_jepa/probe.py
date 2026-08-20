"""Action-blind frozen transition probes and imbalance-robust evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


ProbeView = Literal["current_only", "endpoint", "trajectory"]


def build_probe_features(
    z0_visual: np.ndarray,
    z_future_visual: np.ndarray,
    z0_proprio: np.ndarray,
    z_future_proprio: np.ndarray,
    view: ProbeView = "trajectory",
) -> np.ndarray:
    """Build state/transition features without ever exposing action."""

    z0_visual = np.asarray(z0_visual)
    z_future_visual = np.asarray(z_future_visual)
    z0_proprio = np.asarray(z0_proprio)
    z_future_proprio = np.asarray(z_future_proprio)
    n, k, horizon, _ = z_future_visual.shape
    visual0 = np.broadcast_to(z0_visual[:, None, :], (n, k, z0_visual.shape[-1]))
    proprio0 = np.broadcast_to(z0_proprio[:, None, :], (n, k, z0_proprio.shape[-1]))
    if view == "current_only":
        pieces = [visual0, proprio0]
    elif view == "endpoint":
        pieces = [visual0, z_future_visual[:, :, -1] - visual0, proprio0, z_future_proprio[:, :, -1] - proprio0]
    elif view == "trajectory":
        visual_steps = [visual0]
        proprio_steps = [proprio0]
        previous_visual = visual0
        previous_proprio = proprio0
        for step in range(horizon):
            visual_steps.append(z_future_visual[:, :, step] - previous_visual)
            proprio_steps.append(z_future_proprio[:, :, step] - previous_proprio)
            previous_visual = z_future_visual[:, :, step]
            previous_proprio = z_future_proprio[:, :, step]
        pieces = visual_steps + proprio_steps
    else:
        raise ValueError(view)
    return np.concatenate(pieces, axis=-1).astype(np.float32)


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= left) & (probabilities < right if right < 1.0 else probabilities <= right)
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(ece)


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    predicted = (probabilities >= threshold).astype(np.int8)
    result = {
        "samples": int(len(y_true)),
        "positive_fraction": float(y_true.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "ece_10bin": expected_calibration_error(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }
    result["auroc"] = float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None
    return result


@dataclass
class FrozenTransitionProbe:
    scaler: StandardScaler
    classifier: LogisticRegression
    calibrator: LogisticRegression
    view: ProbeView
    threshold: float = 0.5

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        shape = features.shape[:-1]
        flat = features.reshape(-1, features.shape[-1])
        logits = self.classifier.decision_function(self.scaler.transform(flat)).reshape(-1, 1)
        probabilities = self.calibrator.predict_proba(logits)[:, 1]
        return probabilities.reshape(shape)


def train_frozen_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    view: ProbeView,
    seed: int,
) -> FrozenTransitionProbe:
    train_flat = train_features.reshape(-1, train_features.shape[-1])
    train_y = train_labels.reshape(-1)
    calibration_flat = calibration_features.reshape(-1, calibration_features.shape[-1])
    calibration_y = calibration_labels.reshape(-1)
    scaler = StandardScaler().fit(train_flat)
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
        solver="lbfgs",
    ).fit(scaler.transform(train_flat), train_y)
    calibration_logits = classifier.decision_function(scaler.transform(calibration_flat)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, max_iter=1000, random_state=seed).fit(
        calibration_logits, calibration_y
    )
    return FrozenTransitionProbe(scaler=scaler, classifier=classifier, calibrator=calibrator, view=view)

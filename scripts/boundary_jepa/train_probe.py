"""Train the action-blind probe on GT latents and freeze it before evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import yaml

from boundary_jepa.probe import (
    build_probe_features,
    classification_metrics,
    train_frozen_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/boundary_jepa/pilot.yaml")
    parser.add_argument("--train-sweeps", default="artifacts/pilot_cache/probe_train_sweeps.h5")
    parser.add_argument("--train-features", default="artifacts/pilot_cache/probe_train_features.h5")
    parser.add_argument("--eval-sweeps", default="artifacts/pilot_cache/evaluation_sweeps.h5")
    parser.add_argument("--eval-features", default="artifacts/pilot_cache/evaluation_features.h5")
    parser.add_argument("--output-dir", default="results/pilot")
    return parser.parse_args()


def load_feature_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        return {key: h5[key][:] for key in [
            "z0_visual_pool",
            "z_true_visual_pool",
            "z_pred_visual_pool",
            "z0_proprio",
            "z_true_proprio",
            "z_pred_proprio",
        ]}


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    train_arrays = load_feature_arrays(args.train_features)
    eval_arrays = load_feature_arrays(args.eval_features)
    with h5py.File(args.train_sweeps, "r") as h5:
        train_labels = h5["any_contact"][:].astype(np.int8)
    with h5py.File(args.eval_sweeps, "r") as h5:
        eval_labels = h5["any_contact"][:].astype(np.int8)

    anchor_order = rng.permutation(train_labels.shape[0])
    calibration_count = max(1, int(round(0.25 * len(anchor_order))))
    calibration_anchors = np.sort(anchor_order[:calibration_count])
    fit_anchors = np.sort(anchor_order[calibration_count:])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}

    for view in ["current_only", "endpoint", "trajectory"]:
        gt_train_features = build_probe_features(
            train_arrays["z0_visual_pool"],
            train_arrays["z_true_visual_pool"],
            train_arrays["z0_proprio"],
            train_arrays["z_true_proprio"],
            view=view,
        )
        probe = train_frozen_probe(
            gt_train_features[fit_anchors],
            train_labels[fit_anchors],
            gt_train_features[calibration_anchors],
            train_labels[calibration_anchors],
            view=view,
            seed=seed,
        )
        gt_eval_features = build_probe_features(
            eval_arrays["z0_visual_pool"],
            eval_arrays["z_true_visual_pool"],
            eval_arrays["z0_proprio"],
            eval_arrays["z_true_proprio"],
            view=view,
        )
        pred_eval_features = build_probe_features(
            eval_arrays["z0_visual_pool"],
            eval_arrays["z_pred_visual_pool"],
            eval_arrays["z0_proprio"],
            eval_arrays["z_pred_proprio"],
            view=view,
        )
        calibration_prob = probe.predict_proba(gt_train_features[calibration_anchors])
        gt_eval_prob = probe.predict_proba(gt_eval_features)
        pred_eval_prob = probe.predict_proba(pred_eval_features)
        all_metrics[view] = {
            "calibration_gt": classification_metrics(train_labels[calibration_anchors], calibration_prob),
            "evaluation_gt": classification_metrics(eval_labels, gt_eval_prob),
            "evaluation_predicted": classification_metrics(eval_labels, pred_eval_prob),
        }
        predictions[f"{view}_gt_probability"] = gt_eval_prob.astype(np.float32)
        predictions[f"{view}_pred_probability"] = pred_eval_prob.astype(np.float32)
        joblib.dump(probe, output_dir / f"frozen_{view}_probe.joblib")

    split_metadata = {
        "seed": seed,
        "fit_anchor_indices": fit_anchors.tolist(),
        "calibration_anchor_indices": calibration_anchors.tolist(),
        "evaluation_anchor_count": int(eval_labels.shape[0]),
        "action_blind": True,
        "probe_training_source": "ground_truth_latents_only",
        "fixed_threshold": 0.5,
    }
    (output_dir / "probe_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    (output_dir / "probe_split.json").write_text(json.dumps(split_metadata, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "probe_predictions.npz", **predictions)
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()

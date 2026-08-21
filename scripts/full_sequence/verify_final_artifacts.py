"""Verify the frozen full-sequence episode set without recomputing results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


EXPECTED_SHAPES = {
    "state": (7,),
    "baseline_normalized_actions": (9001, 6, 10),
    "baseline_l2_scores": (9001,),
    "baseline_pred_features": (9001, 400),
    "baseline_pred_readout_scores": (9001,),
    "baseline_gt_readout_scores": (9001,),
    "baseline_simulator_cost": (9001,),
    "baseline_final_states": (9001, 7),
    "baseline_contact_any": (9001,),
    "baseline_gt_features": (9001, 400),
    "baseline_trace_scores": (30, 300),
    "baseline_means_before": (30, 6, 10),
    "baseline_stds_before": (30, 6, 10),
    "baseline_means_after": (30, 6, 10),
    "baseline_stds_after": (30, 6, 10),
    "targeted_normalized_actions": (9001, 6, 10),
    "targeted_pred_features": (9001, 400),
    "targeted_aligned_scores": (9001,),
    "targeted_l2_scores": (9001,),
    "targeted_simulator_cost": (9001,),
    "targeted_final_states": (9001, 7),
    "targeted_contact_any": (9001,),
    "targeted_trace_scores": (30, 300),
    "targeted_means_before": (30, 6, 10),
    "targeted_stds_before": (30, 6, 10),
    "targeted_means_after": (30, 6, 10),
    "targeted_stds_after": (30, 6, 10),
}


def _finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("artifacts/full_sequence/final"))
    parser.add_argument("--output", type=Path, default=Path("results/full_sequence/final_artifact_audit.json"))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--first-seed", type=int, default=20360830)
    args = parser.parse_args()

    failures: list[str] = []
    verified: list[dict[str, int]] = []
    expected_names = {f"episode_{episode:03d}" for episode in range(args.episodes)}
    actual_json = {path.stem for path in args.directory.glob("episode_*.json")}
    actual_npz = {path.stem for path in args.directory.glob("episode_*.npz")}
    if actual_json != expected_names:
        failures.append(f"JSON set mismatch: expected {len(expected_names)}, found {len(actual_json)}")
    if actual_npz != expected_names:
        failures.append(f"NPZ set mismatch: expected {len(expected_names)}, found {len(actual_npz)}")

    for episode in range(args.episodes):
        stem = f"episode_{episode:03d}"
        json_path = args.directory / f"{stem}.json"
        npz_path = args.directory / f"{stem}.npz"
        if not json_path.exists() or not npz_path.exists():
            continue
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        expected_seed = args.first_seed + episode
        if metadata.get("episode") != episode or metadata.get("seed") != expected_seed:
            failures.append(f"{stem}: episode/seed mismatch")
        if not _finite_tree(metadata):
            failures.append(f"{stem}: non-finite JSON value")
        if metadata.get("readout_equivalence_max_abs", float("inf")) > 1e-5:
            failures.append(f"{stem}: readout equivalence exceeds tolerance")
        with np.load(npz_path, allow_pickle=False) as payload:
            if set(payload.files) != set(EXPECTED_SHAPES):
                failures.append(f"{stem}: NPZ key set mismatch")
            for key, expected_shape in EXPECTED_SHAPES.items():
                if key not in payload:
                    continue
                array = payload[key]
                if array.shape != expected_shape:
                    failures.append(f"{stem}:{key}: shape {array.shape} != {expected_shape}")
                if not np.isfinite(array).all():
                    failures.append(f"{stem}:{key}: non-finite value")
        verified.append({"episode": episode, "seed": expected_seed})

    result = {
        "schema_version": 1,
        "passed": not failures and len(verified) == args.episodes,
        "expected_episodes": args.episodes,
        "verified_episodes": len(verified),
        "first_seed": args.first_seed,
        "last_seed": args.first_seed + args.episodes - 1,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

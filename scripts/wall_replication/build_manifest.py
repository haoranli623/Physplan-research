"""Build a checksum manifest for canonical Wall phase artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


FILES = [
    "artifacts/checkpoints/jepa_wm_wall.pth.tar",
    "artifacts/wall_replication/feasibility_features.npz",
    "artifacts/wall_replication/feasibility_features.timing.json",
    "artifacts/wall_replication/development_features.npz",
    "artifacts/wall_replication/development_features.timing.json",
    "artifacts/wall_replication/diagnosis_features.npz",
    "artifacts/wall_replication/diagnosis_features.timing.json",
    "artifacts/wall_replication/repair_features.npz",
    "artifacts/wall_replication/repair_features.timing.json",
    "artifacts/wall_replication/repair_repaired_predictions.npz",
    "artifacts/wall_replication/repair_repaired_predictions.json",
    "configs/wall_replication/feasibility.yaml",
    "configs/wall_replication/development.yaml",
    "configs/wall_replication/protocol.yaml",
    "configs/wall_replication/diagnosis.yaml",
    "configs/wall_replication/repair.yaml",
    "results/wall_replication/feasibility.json",
    "results/wall_replication/development/development_summary.json",
    "results/wall_replication/development/development_scores.npz",
    "results/wall_replication/development/frozen_gt_latent_ridge.joblib",
    "results/wall_replication/development/predictor_repair_development.json",
    "results/wall_replication/development/predictor_repair_dev_selected.pth.tar",
    "results/wall_replication/diagnosis/diagnosis.json",
    "results/wall_replication/diagnosis/state_metrics.csv",
    "results/wall_replication/diagnosis/scores.npz",
    "results/wall_replication/repair/repair.json",
    "results/wall_replication/repair/state_metrics.csv",
    "results/wall_replication/repair/scores.npz",
    "plots/wall_replication/wall_bottleneck_profile.png",
    "plots/wall_replication/wall_bottleneck_profile.pdf",
    "plots/wall_replication/wall_repair_validation.png",
    "plots/wall_replication/wall_repair_validation.pdf",
    "plots/wall_replication/wall_representative_state.png",
    "plots/wall_replication/wall_representative_state.pdf",
    "plots/wall_replication/wall_failure_state.png",
    "plots/wall_replication/wall_failure_state.pdf",
    "plots/wall_replication/selected_states.json",
    "wall_bottleneck_prediction.json",
    "WALL_FEASIBILITY_REPORT.md",
    "WALL_REPLICATION_SUMMARY.md",
    "wall_replication_report.md",
    "WALL_PHASE_SUMMARY.md",
    "docs/wall_replication.md",
    "paper/outline.md",
    "paper/claims_and_evidence.md",
    "paper/related_work_notes.md",
    "paper/figure_plan.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    missing = [value for value in FILES if not Path(value).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing canonical artifacts: {missing}")
    artifacts = [
        {"path": value, "bytes": Path(value).stat().st_size, "sha256": sha256(Path(value))}
        for value in FILES
    ]
    manifest = {
        "schema_version": 1,
        "phase": "wall_independent_replication",
        "initial_pusht_commit": "ad4a2524eb7b36ca59c56cf3478435d290334f2c",
        "protocol_freeze_commit": "71e019f17df524602c9a5db89205da161a4e372a",
        "repair_prediction_commit": "8559594b94846b33e84292b60dfdb308bb927b18",
        "artifacts": artifacts,
    }
    output = Path("results/wall_replication/artifact_manifest.json")
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(artifacts)} entries to {output}")


if __name__ == "__main__":
    main()

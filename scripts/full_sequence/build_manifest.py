"""Hash all protocol, code, result, report, and plot artifacts for this phase."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOTS = [
    Path("configs/full_sequence"),
    Path("artifacts/full_sequence/final"),
    Path("results/full_sequence"),
    Path("plots/full_sequence"),
    Path("scripts/full_sequence"),
]
FILES = [
    Path("boundary_jepa/full_sequence.py"),
    Path("tests/test_full_sequence.py"),
    Path("FULL_SEQUENCE_SUMMARY.md"),
    Path("full_sequence_report.md"),
    Path("full_sequence_status.md"),
    Path("PROJECT_CURRENT_STATUS.md"),
    Path("OVERNIGHT_SUMMARY.md"),
    Path("docs/full_sequence.md"),
    Path("paper/draft.md"),
    Path("paper/claims_and_evidence.md"),
    Path("paper/related_work_notes.md"),
    Path("paper/figure_plan.md"),
    Path("paper/tables.md"),
    Path("paper/limitations.md"),
    Path("paper/outline.md"),
]


def main() -> None:
    manifest_path = Path("results/full_sequence/artifact_manifest.json")
    paths = list(FILES)
    for root in ROOTS:
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file() and path != manifest_path)
    entries = []
    for path in sorted(set(paths), key=lambda x: x.as_posix()):
        if not path.exists():
            continue
        payload = path.read_bytes()
        entries.append({
            "path": path.as_posix(), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest().upper(),
        })
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"schema_version": 1, "files": entries}, indent=2), encoding="utf-8")
    print(f"wrote {len(entries)} entries to {manifest_path}")


if __name__ == "__main__":
    main()

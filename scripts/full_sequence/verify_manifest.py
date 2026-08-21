"""Independently verify the full-sequence SHA-256 artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/full_sequence/artifact_manifest.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["files"]:
        path = Path(entry["path"])
        if not path.exists():
            failures.append({"path": entry["path"], "reason": "missing"})
            continue
        actual_bytes = path.stat().st_size
        actual_sha256 = _sha256(path)
        if actual_bytes != entry["bytes"] or actual_sha256 != entry["sha256"]:
            failures.append({
                "path": entry["path"],
                "reason": "size_or_hash_mismatch",
                "expected_bytes": entry["bytes"],
                "actual_bytes": actual_bytes,
                "expected_sha256": entry["sha256"],
                "actual_sha256": actual_sha256,
            })
    result = {
        "passed": not failures,
        "verified_files": len(manifest["files"]) - len(failures),
        "total_files": len(manifest["files"]),
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

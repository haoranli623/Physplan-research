"""Verify frozen Push-T and Wall manifest entries without modifying them."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


MANIFESTS = [
    Path("results/bottleneck_decomposition/artifact_manifest.json"),
    Path("results/wall_replication/artifact_manifest.json"),
]
AUTHORIZED_MUTABLE = {
    "paper/claims_and_evidence.md",
    "paper/related_work_notes.md",
    "paper/figure_plan.md",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checks = []
    authorized_mutable = []
    for manifest_path in MANIFESTS:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("files", manifest.get("artifacts", []))
        for entry in entries:
            path = Path(entry["path"])
            if path.as_posix() in AUTHORIZED_MUTABLE:
                authorized_mutable.append({
                    "manifest": manifest_path.as_posix(), "path": path.as_posix(),
                    "reason": "paper source explicitly authorized for update in the current phase",
                })
                continue
            payload = path.read_bytes() if path.exists() else b""
            actual_hash = hashlib.sha256(payload).hexdigest().upper() if path.exists() else None
            expected_hash = entry["sha256"].upper()
            expected_size = int(entry.get("bytes", entry.get("size_bytes", -1)))
            checks.append({
                "manifest": manifest_path.as_posix(), "path": path.as_posix(),
                "exists": path.exists(), "expected_bytes": expected_size,
                "actual_bytes": len(payload) if path.exists() else None,
                "expected_sha256": expected_hash, "actual_sha256": actual_hash,
                "passed": path.exists() and len(payload) == expected_size and actual_hash == expected_hash,
            })
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "manifests": [path.as_posix() for path in MANIFESTS],
        "passed": all(item["passed"] for item in checks),
        "passed_count": sum(item["passed"] for item in checks),
        "total_count": len(checks),
        "checks": checks,
        "authorized_mutable_manifest_entries": authorized_mutable,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["passed", "passed_count", "total_count", "git_head"]}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

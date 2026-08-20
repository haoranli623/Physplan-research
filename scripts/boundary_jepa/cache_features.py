"""Cache true and baseline-predicted trajectory features for a sweep split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from boundary_jepa.baseline import load_official_pusht_jepa
from boundary_jepa.features import cache_sweep_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/boundary_jepa/pilot.yaml")
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--no-bfloat16", action="store_true")
    parser.add_argument("--anchor-batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    source = args.source or cfg.get("test", {}).get("sweeps") or cfg.get("source_sweeps")
    output = args.output or cfg.get("test", {}).get("features") or cfg.get("output_features")
    if not source or not output:
        raise ValueError("Provide --source/--output or configure test/source and feature paths")
    bundle = load_official_pusht_jepa(cfg["checkpoint"], device=cfg["device"])
    cache_sweep_features(
        source,
        output,
        bundle,
        np.asarray(cfg["environment"]["canonical_goal_state"], dtype=np.float32),
        use_bfloat16=not args.no_bfloat16,
        anchor_batch_size=args.anchor_batch_size,
    )


if __name__ == "__main__":
    main()

"""Generate Wall cloned candidates and frozen JEPA feature caches."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from wall_replication.baseline import load_official_wall_jepa
from wall_replication.features import cache_wall_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wall_replication/feasibility.yaml")
    parser.add_argument("--anchors", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    split = cfg["split"]
    anchors = args.anchors or int(
        split.get("anchors", split.get("fit_anchors", 0) + split.get("heldout_anchors", 0))
    )
    output = args.output or cfg["outputs"]["cache"]
    bundle = load_official_wall_jepa(cfg["checkpoint"], device=cfg["device"])
    timing = cache_wall_features(
        output,
        bundle,
        seed_start=int(split["feasibility_only_seed_start"]),
        anchors=anchors,
        candidate_count=int(cfg["candidate_set"]["samples"]),
        wall_x=int(cfg["environment"]["wall_x"]),
        door_y=int(cfg["environment"]["door_y"]),
        waypoint_y_min=float(cfg["candidate_set"]["waypoint_y_min"]),
        waypoint_y_max=float(cfg["candidate_set"]["waypoint_y_max"]),
        frameskip=int(cfg["environment"]["frameskip"]),
        horizon=int(cfg["environment"]["latent_horizon"]),
        layouts=[tuple(map(int, layout)) for layout in split.get("layouts", [])] or None,
    )
    print(yaml.safe_dump(timing, sort_keys=True))


if __name__ == "__main__":
    main()

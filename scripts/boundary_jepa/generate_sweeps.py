"""Generate resumable, state-disjoint cloned Push-T action sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from boundary_jepa.sweeps import SweepSpec, generate_sweeps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/boundary_jepa/pilot.yaml")
    parser.add_argument("--split", choices=["probe_train", "evaluation"], required=True)
    parser.add_argument("--anchors", type=int)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    sweep = cfg["sweep"]
    env = cfg["environment"]
    anchors = args.anchors or int(sweep[f"{args.split}_anchors"])
    output = args.output or f"artifacts/pilot_cache/{args.split}_sweeps.h5"
    spec = SweepSpec(
        samples=int(sweep["samples_per_sweep"]),
        angular_half_width=float(sweep["angular_half_width_radians"]),
        action_magnitude=float(sweep["action_magnitude"]),
        min_agent_radius=float(sweep["min_agent_radius"]),
        max_agent_radius=float(sweep["max_agent_radius"]),
        frameskip=int(env["frameskip"]),
        latent_horizon=int(env["latent_horizon"]),
        require_single_boundary=bool(sweep["require_single_boundary"]),
    )
    summary = generate_sweeps(
        output,
        split=args.split,
        anchors=anchors,
        seed=int(cfg["seed"]),
        spec=spec,
        max_attempts_factor=int(sweep["max_anchor_attempts_factor"]),
    )
    summary_path = Path(output).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

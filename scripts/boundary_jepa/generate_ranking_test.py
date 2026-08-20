"""Generate the fresh, simulator-filtered ranking test split after freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from boundary_jepa.sweeps import SweepSpec, generate_sweeps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ranking_diagnostic/protocol.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    candidate = cfg["candidate_set"]
    env = cfg["environment"]
    test = cfg["test"]
    spec = SweepSpec(
        samples=int(candidate["samples"]),
        angular_half_width=float(candidate["angular_half_width_radians"]),
        action_magnitude=float(candidate["action_magnitude"]),
        min_agent_radius=float(candidate["min_agent_radius"]),
        max_agent_radius=float(candidate["max_agent_radius"]),
        frameskip=int(env["frameskip"]),
        latent_horizon=int(env["latent_horizon"]),
        require_single_boundary=bool(candidate["require_single_contact_boundary"]),
        min_task_cost_range=float(candidate["informative_cost_range"]),
    )
    summary = generate_sweeps(
        test["sweeps"],
        split=test["split"],
        anchors=int(test["anchors"]),
        seed=int(cfg["seed"]),
        spec=spec,
        max_attempts_factor=int(candidate["max_attempts_factor"]),
    )
    output = Path(test["sweeps"]).with_suffix(".summary.json")
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

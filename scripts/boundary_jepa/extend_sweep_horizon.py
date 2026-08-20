"""Extend saved anchor/action sweeps to a longer horizon without resampling."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml

from boundary_jepa.pusht import rollout_from_cloned_state
from boundary_jepa.sweeps import SweepSpec, SweepWriter, crossing_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/boundary_jepa/diagnostic_h6.yaml")
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--anchors", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    source_path = args.source or cfg["source_sweeps"]
    output_path = args.output or cfg["output_sweeps"]
    horizon = int(cfg["environment"]["latent_horizon"])
    frameskip = int(cfg["environment"]["frameskip"])

    with h5py.File(source_path, "r") as source:
        source_anchors, samples = source["actions"].shape[:2]
        anchors = args.anchors or source_anchors
        if anchors > source_anchors:
            raise ValueError("Requested more anchors than are present in the source cache")
        spec = SweepSpec(
            samples=samples,
            angular_half_width=float(np.max(np.abs(source["offsets"][0]))),
            action_magnitude=float(np.linalg.norm(source["actions"][0, 0])),
            min_agent_radius=0.0,
            max_agent_radius=0.0,
            frameskip=frameskip,
            latent_horizon=horizon,
            require_single_boundary=False,
        )
        writer = SweepWriter(output_path, anchors, spec, "evaluation_h6_diagnostic", int(cfg["seed"]))
        writer.handle.attrs["source_sweeps"] = str(Path(source_path).resolve())
        writer.handle.attrs["same_anchor_action_extension"] = 1
        rng = np.random.default_rng(int(cfg["seed"]))
        try:
            for anchor in range(writer.completed, anchors):
                state = source["anchor_states"][anchor].astype(np.float64)
                actions = source["actions"][anchor].astype(np.float64)
                offsets = source["offsets"][anchor].astype(np.float64)
                side = int(source["sweep_side"][anchor])
                rollouts = []
                labels = []
                endpoint_steps = set(range(frameskip, frameskip * horizon + 1, frameskip))
                for action in actions:
                    controls = np.repeat(action[None], frameskip * horizon, axis=0)
                    rollout = rollout_from_cloned_state(
                        state,
                        controls,
                        seed=int(cfg["seed"]) + anchor,
                        render_steps=endpoint_steps,
                    )
                    rollouts.append(rollout)
                    labels.append(int(rollout.physical["any_contact"]))
                crossings = crossing_indices(np.asarray(labels, dtype=np.int8))
                writer.write_anchor(
                    state,
                    actions,
                    offsets,
                    side,
                    rollouts,
                    int(crossings[0]) if crossings.size else -1,
                    rng,
                )
                print(
                    f"[same-anchor H{horizon}] completed {writer.completed}/{anchors}; "
                    f"mode crossings={len(crossings)}",
                    flush=True,
                )
        finally:
            writer.close()


if __name__ == "__main__":
    main()

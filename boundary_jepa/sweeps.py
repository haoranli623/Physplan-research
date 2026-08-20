"""One-dimensional local action sweeps with fixed, predeclared acceptance rules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from boundary_jepa.pusht import rollout_from_cloned_state


SPLIT_SEED_OFFSETS = {"probe_train": 0, "evaluation": 1_000_000}


@dataclass(frozen=True)
class SweepSpec:
    samples: int
    angular_half_width: float
    action_magnitude: float
    min_agent_radius: float
    max_agent_radius: float
    frameskip: int
    latent_horizon: int
    require_single_boundary: bool = True

    @property
    def control_steps(self) -> int:
        return self.frameskip * self.latent_horizon


def crossing_indices(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int8)
    return np.flatnonzero(labels[:-1] != labels[1:])


def sample_anchor_and_actions(rng: np.random.Generator, spec: SweepSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sample a near-contact anchor and a one-sided angular action slice."""

    block = rng.uniform([180.0, 180.0], [332.0, 332.0])
    block_angle = rng.uniform(0.0, 2.0 * np.pi)
    agent_angle = rng.uniform(0.0, 2.0 * np.pi)
    radius = rng.uniform(spec.min_agent_radius, spec.max_agent_radius)
    agent = block + radius * np.array([np.cos(agent_angle), np.sin(agent_angle)])
    if np.any(agent < 25.0) or np.any(agent > 487.0):
        raise ValueError("sampled agent outside safe workspace")
    state = np.array([agent[0], agent[1], block[0], block[1], block_angle, 0.0, 0.0], dtype=np.float64)

    toward_block = np.arctan2(block[1] - agent[1], block[0] - agent[0])
    side = 1 if rng.integers(0, 2) else -1
    offsets = np.linspace(0.0, side * spec.angular_half_width, spec.samples, dtype=np.float64)
    angles = toward_block + offsets
    actions = spec.action_magnitude * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    return state, actions, offsets, side


class SweepWriter:
    """Fixed-size, per-anchor flushed HDF5 writer that can resume safely."""

    def __init__(self, path: str | Path, anchors: int, spec: SweepSpec, split: str, seed: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.anchors = anchors
        self.spec = spec
        self.split = split
        self.seed = seed
        self.handle = h5py.File(self.path, "a")
        if "anchor_states" not in self.handle:
            self._initialize()
        self._validate_metadata()

    def _initialize(self) -> None:
        h5 = self.handle
        n, k, h, t = self.anchors, self.spec.samples, self.spec.latent_horizon, self.spec.control_steps
        h5.attrs.update(
            {
                "schema_version": 1,
                "split": self.split,
                "seed": self.seed,
                "completed_anchors": 0,
                "attempted_anchors": 0,
                "rejected_no_boundary": 0,
                "rejected_multi_boundary": 0,
                "rejected_workspace": 0,
                "frameskip": self.spec.frameskip,
                "latent_horizon": h,
                "samples_per_sweep": k,
                "require_single_boundary": int(self.spec.require_single_boundary),
            }
        )
        # Exact float64 anchor/action values are required for bitwise-reproducible
        # cloned contact dynamics. Float32 truncation can be amplified at impact.
        h5.create_dataset("anchor_states", (n, 7), dtype="f8")
        h5.create_dataset("initial_images", (n, 224, 224, 3), dtype="u1", compression="lzf")
        h5.create_dataset("actions", (n, k, 2), dtype="f8")
        h5.create_dataset("offsets", (n, k), dtype="f8")
        h5.create_dataset("sweep_side", (n,), dtype="i1")
        h5.create_dataset("future_images", (n, k, h, 224, 224, 3), dtype="u1", compression="lzf")
        h5.create_dataset("future_states", (n, k, h, 7), dtype="f4")
        h5.create_dataset("contact_steps", (n, k, t), dtype="i1")
        h5.create_dataset("any_contact", (n, k), dtype="i1")
        h5.create_dataset("contact_fraction", (n, k), dtype="f4")
        h5.create_dataset("persistent_contact", (n, k), dtype="i1")
        h5.create_dataset("lost_contact", (n, k), dtype="i1")
        h5.create_dataset("final_coverage", (n, k), dtype="f4")
        h5.create_dataset("coverage_progress", (n, k), dtype="f4")
        h5.create_dataset("task_success", (n, k), dtype="i1")
        h5.create_dataset("task_cost", (n, k), dtype="f4")
        h5.create_dataset("true_crossing_index", (n,), dtype="i2", fillvalue=-1)
        h5.flush()

    def _validate_metadata(self) -> None:
        h5 = self.handle
        expected = {
            "split": self.split,
            "seed": self.seed,
            "frameskip": self.spec.frameskip,
            "latent_horizon": self.spec.latent_horizon,
            "samples_per_sweep": self.spec.samples,
        }
        for key, value in expected.items():
            if h5.attrs[key] != value:
                raise ValueError(f"Cannot resume {self.path}: {key}={h5.attrs[key]!r}, expected {value!r}")
        if h5["anchor_states"].shape[0] != self.anchors:
            raise ValueError("Requested anchor count differs from existing cache")

    @property
    def completed(self) -> int:
        return int(self.handle.attrs["completed_anchors"])

    def write_anchor(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        offsets: np.ndarray,
        side: int,
        rollouts: list,
        crossing_index: int,
        rng: np.random.Generator,
    ) -> None:
        index = self.completed
        h5 = self.handle
        h5["anchor_states"][index] = state
        h5["initial_images"][index] = rollouts[0].images[0]
        h5["actions"][index] = actions
        h5["offsets"][index] = offsets
        h5["sweep_side"][index] = side
        endpoints = np.arange(self.spec.frameskip, self.spec.control_steps + 1, self.spec.frameskip)
        for j, rollout in enumerate(rollouts):
            h5["future_images"][index, j] = rollout.images[endpoints]
            h5["future_states"][index, j] = rollout.states[endpoints]
            h5["contact_steps"][index, j] = rollout.contacts
            h5["any_contact"][index, j] = rollout.physical["any_contact"]
            h5["contact_fraction"][index, j] = rollout.physical["contact_fraction"]
            h5["persistent_contact"][index, j] = rollout.physical["persistent_contact"]
            h5["lost_contact"][index, j] = rollout.physical["lost_contact"]
            h5["final_coverage"][index, j] = rollout.task["final_coverage"]
            h5["coverage_progress"][index, j] = rollout.task["coverage_progress"]
            h5["task_success"][index, j] = rollout.task["success"]
            h5["task_cost"][index, j] = rollout.task["cost"]
        h5["true_crossing_index"][index] = crossing_index
        h5.attrs["rng_state_json"] = json.dumps(rng.bit_generator.state)
        h5.attrs["completed_anchors"] = index + 1
        h5.flush()

    def close(self) -> None:
        self.handle.close()


def generate_sweeps(
    path: str | Path,
    *,
    split: str,
    anchors: int,
    seed: int,
    spec: SweepSpec,
    max_attempts_factor: int,
) -> dict[str, int]:
    if split not in SPLIT_SEED_OFFSETS:
        raise ValueError(split)
    effective_seed = int(seed + SPLIT_SEED_OFFSETS[split])
    writer = SweepWriter(path, anchors, spec, split, effective_seed)
    rng = np.random.default_rng(effective_seed)
    if writer.completed:
        state_json = writer.handle.attrs.get("rng_state_json")
        if state_json:
            rng.bit_generator.state = json.loads(state_json)
    try:
        max_attempts = anchors * max_attempts_factor
        while writer.completed < anchors and int(writer.handle.attrs["attempted_anchors"]) < max_attempts:
            writer.handle.attrs.modify("attempted_anchors", int(writer.handle.attrs["attempted_anchors"]) + 1)
            try:
                state, candidate_actions, offsets, side = sample_anchor_and_actions(rng, spec)
            except ValueError:
                writer.handle.attrs.modify("rejected_workspace", int(writer.handle.attrs["rejected_workspace"]) + 1)
                continue

            rollouts = []
            labels = []
            for candidate_index, action in enumerate(candidate_actions):
                control_actions = np.repeat(action[None], spec.control_steps, axis=0)
                rollout = rollout_from_cloned_state(
                    state,
                    control_actions,
                    seed=effective_seed + int(writer.handle.attrs["attempted_anchors"]),
                )
                rollouts.append(rollout)
                labels.append(int(rollout.physical["any_contact"]))
            crossings = crossing_indices(np.asarray(labels))
            if len(crossings) == 0:
                writer.handle.attrs.modify("rejected_no_boundary", int(writer.handle.attrs["rejected_no_boundary"]) + 1)
                continue
            if spec.require_single_boundary and len(crossings) != 1:
                writer.handle.attrs.modify(
                    "rejected_multi_boundary", int(writer.handle.attrs["rejected_multi_boundary"]) + 1
                )
                continue
            writer.write_anchor(state, candidate_actions, offsets, side, rollouts, int(crossings[0]), rng)
            print(
                f"[{split}] accepted {writer.completed}/{anchors}; "
                f"attempts={int(writer.handle.attrs['attempted_anchors'])}",
                flush=True,
            )
        if writer.completed < anchors:
            raise RuntimeError(f"Only generated {writer.completed}/{anchors} anchors within {max_attempts} attempts")
        return {key: int(writer.handle.attrs[key]) for key in [
            "completed_anchors",
            "attempted_anchors",
            "rejected_no_boundary",
            "rejected_multi_boundary",
            "rejected_workspace",
        ]}
    finally:
        writer.close()

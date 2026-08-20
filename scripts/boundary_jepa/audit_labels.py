"""Audit simulator-native contact labels before fitting any probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from boundary_jepa.pusht import rollout_from_cloned_state
from boundary_jepa.sweeps import crossing_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps", default="artifacts/pilot_cache/evaluation_sweeps.h5")
    parser.add_argument("--output", default="results/pilot/label_audit.json")
    parser.add_argument("--plot", default="plots/pilot/label_audit.png")
    parser.add_argument("--replays", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with h5py.File(args.sweeps, "r") as h5:
        completed = int(h5.attrs["completed_anchors"])
        states = h5["anchor_states"][:]
        actions = h5["actions"][:]
        labels = h5["any_contact"][:].astype(np.int8)
        contacts = h5["contact_steps"][:].astype(np.int8)
        future_states = h5["future_states"][:]
        final_coverage = h5["final_coverage"][:]
        future_images = h5["future_images"]
        frameskip = int(h5.attrs["frameskip"])
        horizon = int(h5.attrs["latent_horizon"])
        split_seed = int(h5.attrs["seed"])

        crossing_counts = np.asarray([len(crossing_indices(row)) for row in labels])
        crossing_locations = np.asarray(
            [crossing_indices(row)[0] for row in labels if len(crossing_indices(row))], dtype=np.int64
        )
        rng = np.random.default_rng(20260820)
        replay_pairs = np.column_stack(
            [
                rng.integers(0, completed, args.replays),
                rng.integers(0, labels.shape[1], args.replays),
            ]
        )
        replay_failures = []
        for anchor, candidate in replay_pairs:
            control_actions = np.repeat(actions[anchor, candidate][None], frameskip * horizon, axis=0)
            replay = rollout_from_cloned_state(
                states[anchor],
                control_actions,
                seed=split_seed + anchor + 1,
            )
            endpoints = np.arange(frameskip, frameskip * horizon + 1, frameskip)
            if not np.array_equal(replay.contacts, contacts[anchor, candidate]):
                replay_failures.append({"anchor": int(anchor), "candidate": int(candidate), "field": "contacts"})
            if not np.allclose(replay.states[endpoints], future_states[anchor, candidate], atol=1e-6, rtol=0):
                replay_failures.append({"anchor": int(anchor), "candidate": int(candidate), "field": "states"})
            if not np.isclose(replay.task["final_coverage"], final_coverage[anchor, candidate], atol=1e-6):
                replay_failures.append({"anchor": int(anchor), "candidate": int(candidate), "field": "coverage"})

        selected = np.linspace(0, labels.shape[1] - 1, 6, dtype=int)
        fig, axes = plt.subplots(2, len(selected), figsize=(15, 5))
        for column, candidate in enumerate(selected):
            axes[0, column].imshow(future_images[0, candidate, 0])
            axes[1, column].imshow(future_images[0, candidate, -1])
            axes[0, column].set_title(f"a={candidate}, contact={labels[0, candidate]}")
            axes[1, column].set_title(f"contact steps={contacts[0, candidate].sum()}")
            axes[0, column].axis("off")
            axes[1, column].axis("off")
        axes[0, 0].set_ylabel("latent step 1")
        axes[1, 0].set_ylabel(f"latent step {horizon}")
        fig.tight_layout()
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    audit = {
        "completed_anchors": completed,
        "actions_per_anchor": int(labels.shape[1]),
        "positive_contact_fraction": float(labels.mean()),
        "all_anchors_have_exactly_one_crossing": bool(np.all(crossing_counts == 1)),
        "first_action_contact_fraction": float(labels[:, 0].mean()),
        "last_action_contact_fraction": float(labels[:, -1].mean()),
        "crossing_index_min": int(crossing_locations.min()) if crossing_locations.size else None,
        "crossing_index_median": float(np.median(crossing_locations)) if crossing_locations.size else None,
        "crossing_index_max": int(crossing_locations.max()) if crossing_locations.size else None,
        "contact_step_fraction": float(contacts.mean()),
        "replay_checks": int(args.replays),
        "replay_failure_count": len(replay_failures),
        "replay_failures": replay_failures,
        "primary_label": f"any simulator collision point during {frameskip * horizon} control steps",
        "task_outcome_used_in_label": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

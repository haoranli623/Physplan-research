"""Validate official checkpoint loading, cloning, labels, and one predictor step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from boundary_jepa.baseline import flatten_normalized_action_chunks, load_official_pusht_jepa
from boundary_jepa.pusht import rollout_from_cloned_state, state_to_proprio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/jepa_wm_pusht.pth.tar")
    parser.add_argument("--output", default="results/baseline_smoke.json")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = np.array([256, 400, 256, 300, 0, 0, 0], dtype=np.float64)
    raw_action = np.array([0.0, -0.5], dtype=np.float64)
    actions = np.repeat(raw_action[None], 5, axis=0)
    rollout1 = rollout_from_cloned_state(state, actions, seed=17)
    rollout2 = rollout_from_cloned_state(state, actions, seed=17)

    state_max_abs = float(np.max(np.abs(rollout1.states - rollout2.states)))
    image_max_abs = int(np.max(np.abs(rollout1.images.astype(np.int16) - rollout2.images.astype(np.int16))))
    contacts_equal = bool(np.array_equal(rollout1.contacts, rollout2.contacts))
    if state_max_abs > 1e-6 or image_max_abs != 0 or not contacts_equal:
        raise RuntimeError("Cloned-state replay is not deterministic")

    bundle = load_official_pusht_jepa(args.checkpoint, device=args.device)
    model = bundle.model
    device = model.device
    initial_image = torch.from_numpy(rollout1.images[0]).permute(2, 0, 1)[None, None]
    final_image = torch.from_numpy(rollout1.images[-1]).permute(2, 0, 1)[None, None]
    proprios = state_to_proprio(rollout1.states)
    initial_obs = TensorDict(
        {
            "visual": initial_image,
            "proprio": torch.from_numpy(proprios[0])[None, None],
        },
        batch_size=[1, 1],
    )
    final_obs = TensorDict(
        {
            "visual": final_image,
            "proprio": torch.from_numpy(proprios[-1])[None, None],
        },
        batch_size=[1, 1],
    )
    with torch.inference_mode():
        z0 = model.encode(initial_obs.to(device))
        z_true = model.encode(final_obs.to(device))
        action_tensor = torch.from_numpy(actions.astype(np.float32))[None]
        action_chunks = flatten_normalized_action_chunks(action_tensor, bundle.preprocessor, frameskip=5).to(device)
        z_unroll = model.unroll(z0, action_chunks)
        z_pred = z_unroll["visual"][-1]
        latent_mse = float(torch.mean((z_pred - z_true["visual"]) ** 2).cpu())

    result = {
        "checkpoint": str(bundle.checkpoint),
        "checkpoint_epoch": 50,
        "device": str(device),
        "clone_state_max_abs_error": state_max_abs,
        "clone_image_max_abs_error": image_max_abs,
        "clone_contacts_equal": contacts_equal,
        "contact_sequence": rollout1.contacts.tolist(),
        "physical": rollout1.physical,
        "task": rollout1.task,
        "z0_visual_shape": list(z0["visual"].shape),
        "z_true_visual_shape": list(z_true["visual"].shape),
        "z_pred_visual_shape": list(z_pred.shape),
        "model_action_shape": list(action_chunks.shape),
        "single_chunk_latent_mse": latent_mse,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

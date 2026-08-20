"""GPU feature caching for cloned-state sweeps using the frozen official model."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
from tensordict import TensorDict

from boundary_jepa.baseline import BaselineBundle, flatten_normalized_action_chunks
from boundary_jepa.pusht import rollout_from_cloned_state, state_to_proprio


def _obs_tensordict(images: np.ndarray, states: np.ndarray) -> TensorDict:
    """Convert [B,T,H,W,C] images and [B,T,7] states to official model input."""

    visual = torch.from_numpy(np.asarray(images)).permute(0, 1, 4, 2, 3).contiguous()
    proprio = torch.from_numpy(state_to_proprio(np.asarray(states)))
    return TensorDict({"visual": visual, "proprio": proprio}, batch_size=list(visual.shape[:2]))


def _pool_visual(features: torch.Tensor) -> torch.Tensor:
    return features.mean(dim=(2, 3, 4))


def _pool_proprio(features: torch.Tensor) -> torch.Tensor:
    """Undo feature-mode spatial repetition while preserving the 16-D embedding."""

    spatial_or_token_dims = tuple(range(2, features.ndim - 1))
    return features.mean(dim=spatial_or_token_dims) if spatial_or_token_dims else features


class FeatureWriter:
    def __init__(self, output: str | Path, source: h5py.File, embed_dim: int, proprio_dim: int):
        self.path = Path(output)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.path, "a")
        n, k = source["any_contact"].shape
        h = int(source.attrs["latent_horizon"])
        if "z0_visual_pool" not in self.handle:
            out = self.handle
            out.attrs.update(
                {
                    "schema_version": 1,
                    "source": str(Path(source.filename).resolve()),
                    "split": source.attrs["split"],
                    "completed_anchors": 0,
                    "embed_dim": embed_dim,
                    "proprio_embed_dim": proprio_dim,
                }
            )
            out.create_dataset("z0_visual_pool", (n, embed_dim), dtype="f4")
            out.create_dataset("z_true_visual_pool", (n, k, h, embed_dim), dtype="f4")
            out.create_dataset("z_pred_visual_pool", (n, k, h, embed_dim), dtype="f4")
            out.create_dataset("z0_proprio", (n, proprio_dim), dtype="f4")
            out.create_dataset("z_true_proprio", (n, k, h, proprio_dim), dtype="f4")
            out.create_dataset("z_pred_proprio", (n, k, h, proprio_dim), dtype="f4")
            out.create_dataset("latent_mse", (n, k, h), dtype="f4")
            out.create_dataset("predicted_goal_cost", (n, k), dtype="f4")
            out.create_dataset("true_latent_goal_cost", (n, k), dtype="f4")
            out.flush()
        if self.handle["z_true_visual_pool"].shape[:3] != (n, k, h):
            raise ValueError("Feature cache shape does not match sweep cache")

    @property
    def completed(self) -> int:
        return int(self.handle.attrs["completed_anchors"])

    def close(self) -> None:
        self.handle.close()


def cache_sweep_features(
    source_path: str | Path,
    output_path: str | Path,
    bundle: BaselineBundle,
    canonical_goal_state: np.ndarray,
    *,
    use_bfloat16: bool = True,
) -> None:
    model = bundle.model
    device = model.device
    with h5py.File(source_path, "r") as source:
        if int(source.attrs["completed_anchors"]) != source["anchor_states"].shape[0]:
            raise RuntimeError("Sweep generation is incomplete")
        frameskip = int(source.attrs["frameskip"])
        horizon = int(source.attrs["latent_horizon"])
        candidates = source["actions"].shape[1]

        goal_rollout = rollout_from_cloned_state(
            np.asarray(canonical_goal_state, dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            seed=0,
        )
        goal_images = goal_rollout.images[None]
        goal_states = goal_rollout.states[None]
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bfloat16 and device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), amp_context:
            z_goal = model.encode(_obs_tensordict(goal_images, goal_states).to(device))
        proprio_dim = int(_pool_proprio(z_goal["proprio"]).shape[-1])
        writer = FeatureWriter(output_path, source, embed_dim=384, proprio_dim=proprio_dim)
        writer.handle.attrs["autocast_bfloat16"] = int(use_bfloat16 and device.type == "cuda")
        writer.handle.attrs["canonical_goal_state"] = canonical_goal_state.astype(np.float32)
        try:
            for anchor_index in range(writer.completed, source["anchor_states"].shape[0]):
                initial_images = source["initial_images"][anchor_index][None, None]
                initial_states = source["anchor_states"][anchor_index][None, None]
                future_images = source["future_images"][anchor_index]
                future_states = source["future_states"][anchor_index]
                raw_action = torch.from_numpy(source["actions"][anchor_index]).float()
                raw_control = raw_action[:, None, :].repeat(1, frameskip * horizon, 1)
                action_chunks = flatten_normalized_action_chunks(
                    raw_control, bundle.preprocessor, frameskip=frameskip
                ).to(device)

                with torch.inference_mode(), amp_context:
                    z0 = model.encode(_obs_tensordict(initial_images, initial_states).to(device))
                    z_true = model.encode(_obs_tensordict(future_images, future_states).to(device))
                    z0_candidates = TensorDict(
                        {
                            "visual": z0["visual"].expand(candidates, *z0["visual"].shape[1:]),
                            "proprio": z0["proprio"].expand(candidates, *z0["proprio"].shape[1:]),
                        },
                        batch_size=[candidates, z0["visual"].shape[1]],
                    )
                    z_pred_unroll = model.unroll(z0_candidates, action_chunks)
                    pred_visual = z_pred_unroll["visual"][-horizon:].transpose(0, 1)
                    pred_proprio = z_pred_unroll["proprio"][-horizon:].transpose(0, 1)
                    true_visual = z_true["visual"]
                    true_proprio = z_true["proprio"]
                    latent_mse = (pred_visual - true_visual).float().pow(2).mean(dim=(2, 3, 4, 5))

                    target_visual = z_goal["visual"][:, -1]
                    target_proprio = z_goal["proprio"][:, -1]
                    pred_goal_cost = (pred_visual[:, -1] - target_visual).float().pow(2).mean(dim=(1, 2, 3, 4))
                    pred_goal_cost += 0.1 * (
                        (pred_proprio[:, -1] - target_proprio).float().pow(2).mean(dim=tuple(range(1, pred_proprio.ndim - 1)))
                    )
                    true_goal_cost = (true_visual[:, -1] - target_visual).float().pow(2).mean(dim=(1, 2, 3, 4))
                    true_goal_cost += 0.1 * (
                        (true_proprio[:, -1] - target_proprio).float().pow(2).mean(dim=tuple(range(1, true_proprio.ndim - 1)))
                    )

                out = writer.handle
                out["z0_visual_pool"][anchor_index] = _pool_visual(z0["visual"])[0, 0].float().cpu().numpy()
                out["z_true_visual_pool"][anchor_index] = _pool_visual(true_visual).float().cpu().numpy()
                out["z_pred_visual_pool"][anchor_index] = _pool_visual(pred_visual).float().cpu().numpy()
                out["z0_proprio"][anchor_index] = _pool_proprio(z0["proprio"])[0, 0].float().cpu().numpy()
                out["z_true_proprio"][anchor_index] = _pool_proprio(true_proprio).float().cpu().numpy()
                out["z_pred_proprio"][anchor_index] = _pool_proprio(pred_proprio).float().cpu().numpy()
                out["latent_mse"][anchor_index] = latent_mse.cpu().numpy()
                out["predicted_goal_cost"][anchor_index] = pred_goal_cost.cpu().numpy()
                out["true_latent_goal_cost"][anchor_index] = true_goal_cost.cpu().numpy()
                out.attrs.modify("completed_anchors", anchor_index + 1)
                out.flush()
                print(
                    f"[{source.attrs['split']}] encoded {anchor_index + 1}/{source['anchor_states'].shape[0]}",
                    flush=True,
                )
        finally:
            writer.close()

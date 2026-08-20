"""GPU feature caching for cloned-state sweeps using the frozen official model."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import time

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
    anchor_batch_size: int = 1,
) -> None:
    if anchor_batch_size < 1:
        raise ValueError("anchor_batch_size must be positive")
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
        writer.handle.attrs["requested_anchor_batch_size"] = int(anchor_batch_size)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        try:
            current_batch_size = int(anchor_batch_size)
            oom_backoffs = 0
            total_anchors = source["anchor_states"].shape[0]
            while writer.completed < total_anchors:
                start = writer.completed
                end = min(start + current_batch_size, total_anchors)
                anchor_count = end - start
                try:
                    initial_images = source["initial_images"][start:end][:, None]
                    initial_states = source["anchor_states"][start:end][:, None]
                    future_images = source["future_images"][start:end].reshape(
                        anchor_count * candidates, horizon, 224, 224, 3
                    )
                    future_states = source["future_states"][start:end].reshape(
                        anchor_count * candidates, horizon, 7
                    )
                    raw_action = torch.from_numpy(source["actions"][start:end]).float().reshape(
                        anchor_count * candidates, 2
                    )
                    raw_control = raw_action[:, None, :].repeat(1, frameskip * horizon, 1)
                    action_chunks = flatten_normalized_action_chunks(
                        raw_control, bundle.preprocessor, frameskip=frameskip
                    ).to(device)

                    with torch.inference_mode(), amp_context:
                        z0 = model.encode(_obs_tensordict(initial_images, initial_states).to(device))
                        z_true = model.encode(_obs_tensordict(future_images, future_states).to(device))
                        z0_candidates = TensorDict(
                            {
                                "visual": z0["visual"].repeat_interleave(candidates, dim=0),
                                "proprio": z0["proprio"].repeat_interleave(candidates, dim=0),
                            },
                            batch_size=[anchor_count * candidates, z0["visual"].shape[1]],
                        )
                        z_pred_unroll = model.unroll(z0_candidates, action_chunks)
                        pred_visual = z_pred_unroll["visual"][-horizon:].transpose(0, 1)
                        pred_proprio = z_pred_unroll["proprio"][-horizon:].transpose(0, 1)
                        true_visual = z_true["visual"]
                        true_proprio = z_true["proprio"]
                        latent_mse = (pred_visual - true_visual).float().pow(2).mean(dim=(2, 3, 4, 5))

                        target_visual = z_goal["visual"][:, -1]
                        target_proprio = z_goal["proprio"][:, -1]
                        pred_goal_cost = (pred_visual[:, -1] - target_visual).float().pow(2).mean(
                            dim=(1, 2, 3, 4)
                        )
                        pred_goal_cost += 0.1 * (
                            (pred_proprio[:, -1] - target_proprio)
                            .float()
                            .pow(2)
                            .mean(dim=tuple(range(1, pred_proprio.ndim - 1)))
                        )
                        true_goal_cost = (true_visual[:, -1] - target_visual).float().pow(2).mean(
                            dim=(1, 2, 3, 4)
                        )
                        true_goal_cost += 0.1 * (
                            (true_proprio[:, -1] - target_proprio)
                            .float()
                            .pow(2)
                            .mean(dim=tuple(range(1, true_proprio.ndim - 1)))
                        )
                except torch.OutOfMemoryError:
                    if current_batch_size == 1:
                        raise
                    oom_backoffs += 1
                    current_batch_size = max(1, current_batch_size // 2)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    print(f"CUDA OOM; backing off to {current_batch_size} anchors per batch", flush=True)
                    continue

                out = writer.handle
                out["z0_visual_pool"][start:end] = _pool_visual(z0["visual"])[:, 0].float().cpu().numpy()
                out["z_true_visual_pool"][start:end] = (
                    _pool_visual(true_visual).reshape(anchor_count, candidates, horizon, -1).float().cpu().numpy()
                )
                out["z_pred_visual_pool"][start:end] = (
                    _pool_visual(pred_visual).reshape(anchor_count, candidates, horizon, -1).float().cpu().numpy()
                )
                out["z0_proprio"][start:end] = _pool_proprio(z0["proprio"])[:, 0].float().cpu().numpy()
                out["z_true_proprio"][start:end] = (
                    _pool_proprio(true_proprio).reshape(anchor_count, candidates, horizon, -1).float().cpu().numpy()
                )
                out["z_pred_proprio"][start:end] = (
                    _pool_proprio(pred_proprio).reshape(anchor_count, candidates, horizon, -1).float().cpu().numpy()
                )
                out["latent_mse"][start:end] = latent_mse.reshape(anchor_count, candidates, horizon).cpu().numpy()
                out["predicted_goal_cost"][start:end] = pred_goal_cost.reshape(anchor_count, candidates).cpu().numpy()
                out["true_latent_goal_cost"][start:end] = true_goal_cost.reshape(anchor_count, candidates).cpu().numpy()
                out.attrs.modify("completed_anchors", end)
                out.attrs["effective_anchor_batch_size"] = int(current_batch_size)
                out.attrs["oom_backoffs"] = int(oom_backoffs)
                out.flush()
                print(f"[{source.attrs['split']}] encoded {end}/{total_anchors}", flush=True)
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                writer.handle.attrs["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
                writer.handle.attrs["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
            elapsed = time.perf_counter() - started
            writer.handle.attrs["feature_cache_seconds"] = float(elapsed)
            completed_candidates = writer.completed * candidates
            writer.handle.attrs["candidate_rollouts_per_second"] = float(completed_candidates / elapsed)
            writer.handle.flush()
            writer.close()

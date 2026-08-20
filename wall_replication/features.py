"""Wall simulator and frozen JEPA feature caching."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import json
import time

import numpy as np
import torch
from tensordict import TensorDict

from wall_replication.baseline import WallBaselineBundle, flatten_normalized_action_chunks
from wall_replication.environment import (
    make_wall_env,
    render_states_224,
    simulate_candidate_set,
    waypoint_candidate_set,
)


def _obs_tensordict(images: np.ndarray, states: np.ndarray) -> TensorDict:
    visual = torch.from_numpy(np.asarray(images)).permute(0, 1, 4, 2, 3).contiguous()
    proprio = torch.from_numpy(np.asarray(states, dtype=np.float32))
    return TensorDict({"visual": visual, "proprio": proprio}, batch_size=list(visual.shape[:2]))


def _pool_visual(features: torch.Tensor) -> torch.Tensor:
    return features.mean(dim=(2, 3, 4))


def _pool_proprio(features: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(2, features.ndim - 1))
    return features.mean(dim=dims) if dims else features


def cache_wall_features(
    output_path: str | Path,
    bundle: WallBaselineBundle,
    *,
    seed_start: int,
    anchors: int,
    candidate_count: int,
    wall_x: int,
    door_y: int,
    waypoint_y_min: float,
    waypoint_y_max: float,
    frameskip: int,
    horizon: int,
    layouts: list[tuple[int, int]] | None = None,
    use_bfloat16: bool = True,
) -> dict[str, float | int]:
    """Generate cloned candidates and cache pooled true/predicted latents."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    controls = frameskip * horizon
    model = bundle.model
    device = model.device
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bfloat16 and device.type == "cuda"
        else nullcontext()
    )

    starts = np.empty((anchors, 2), dtype=np.float32)
    goals = np.empty((anchors, 2), dtype=np.float32)
    waypoint_y = np.empty((anchors, candidate_count), dtype=np.float32)
    actions = np.empty((anchors, candidate_count, controls, 2), dtype=np.float32)
    sampled_states = np.empty((anchors, candidate_count, horizon + 1, 2), dtype=np.float32)
    collisions = np.empty((anchors, candidate_count, controls), dtype=bool)
    simulator_cost = np.empty((anchors, candidate_count), dtype=np.float32)
    crossed_wall = np.empty((anchors, candidate_count), dtype=bool)
    wall_x_values = np.empty(anchors, dtype=np.int16)
    door_y_values = np.empty(anchors, dtype=np.int16)
    true_visual = np.empty((anchors, candidate_count, horizon, 384), dtype=np.float32)
    pred_visual = np.empty_like(true_visual)
    true_proprio = np.empty((anchors, candidate_count, horizon, 16), dtype=np.float32)
    pred_proprio = np.empty_like(true_proprio)
    start_visual = np.empty((anchors, 384), dtype=np.float32)
    start_proprio = np.empty((anchors, 16), dtype=np.float32)
    goal_visual = np.empty((anchors, 384), dtype=np.float32)
    goal_proprio = np.empty((anchors, 16), dtype=np.float32)
    default_true_score = np.empty((anchors, candidate_count), dtype=np.float32)
    default_pred_score = np.empty_like(default_true_score)
    latent_mse = np.empty_like(default_true_score)

    simulator_seconds = 0.0
    render_seconds = 0.0
    encoder_seconds = 0.0
    predictor_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for anchor in range(anchors):
        state_seed = seed_start + anchor
        anchor_wall_x, anchor_door_y = (
            layouts[anchor % len(layouts)] if layouts else (wall_x, door_y)
        )
        env = make_wall_env(wall_x=anchor_wall_x, door_y=anchor_door_y, seed=state_seed)
        start, goal = env.generate_random_state(seed=state_seed)
        ys, candidate_actions = waypoint_candidate_set(
            start,
            goal,
            wall_x=anchor_wall_x,
            candidate_count=candidate_count,
            y_min=waypoint_y_min,
            y_max=waypoint_y_max,
            control_steps=controls,
        )
        tick = time.perf_counter()
        rollout = simulate_candidate_set(
            start,
            candidate_actions,
            wall_x=anchor_wall_x,
            door_y=anchor_door_y,
            frameskip=frameskip,
            seed=state_seed,
        )
        simulator_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        initial_image = render_states_224(
            start[None], wall_x=anchor_wall_x, door_y=anchor_door_y
        )[None]
        goal_image = render_states_224(
            goal[None], wall_x=anchor_wall_x, door_y=anchor_door_y
        )[None]
        future_images = render_states_224(
            rollout.sampled_states[:, 1:], wall_x=anchor_wall_x, door_y=anchor_door_y
        )
        render_seconds += time.perf_counter() - tick

        initial_states = np.asarray(start, dtype=np.float32)[None, None]
        goal_states = np.asarray(goal, dtype=np.float32)[None, None]
        chunks = flatten_normalized_action_chunks(
            torch.from_numpy(candidate_actions), bundle.preprocessor, frameskip=frameskip
        ).to(device)
        with torch.inference_mode(), amp_context:
            tick = time.perf_counter()
            z0 = model.encode(_obs_tensordict(initial_image, initial_states).to(device))
            zg = model.encode(_obs_tensordict(goal_image, goal_states).to(device))
            zt = model.encode(
                _obs_tensordict(future_images, rollout.sampled_states[:, 1:]).to(device)
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            encoder_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            zp_unroll = model.unroll(z0, chunks)
            zp_visual = zp_unroll["visual"][-horizon:].transpose(0, 1)
            zp_proprio = zp_unroll["proprio"][-horizon:].transpose(0, 1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            predictor_seconds += time.perf_counter() - tick

            zt_visual = zt["visual"]
            zt_proprio = zt["proprio"]
            goal_v = zg["visual"][:, -1]
            goal_p = zg["proprio"][:, -1]
            true_score = (zt_visual[:, -1] - goal_v).float().pow(2).mean(dim=(1, 2, 3, 4))
            true_score += 0.1 * (zt_proprio[:, -1] - goal_p).float().pow(2).mean(
                dim=tuple(range(1, zt_proprio.ndim - 1))
            )
            pred_score = (zp_visual[:, -1] - goal_v).float().pow(2).mean(dim=(1, 2, 3, 4))
            pred_score += 0.1 * (zp_proprio[:, -1] - goal_p).float().pow(2).mean(
                dim=tuple(range(1, zp_proprio.ndim - 1))
            )
            mse = (zp_visual - zt_visual).float().pow(2).mean(dim=(1, 2, 3, 4, 5))

        starts[anchor] = start
        goals[anchor] = goal
        waypoint_y[anchor] = ys
        actions[anchor] = candidate_actions
        sampled_states[anchor] = rollout.sampled_states
        collisions[anchor] = rollout.collision_steps
        simulator_cost[anchor] = np.linalg.norm(rollout.sampled_states[:, -1] - goal, axis=1)
        wall_x_values[anchor] = anchor_wall_x
        door_y_values[anchor] = anchor_door_y
        goal_side = np.sign(goal[0] - anchor_wall_x)
        crossed_wall[anchor] = (
            np.sign(rollout.sampled_states[:, -1, 0] - anchor_wall_x) == goal_side
        )
        start_visual[anchor] = _pool_visual(z0["visual"])[0, 0].float().cpu().numpy()
        start_proprio[anchor] = _pool_proprio(z0["proprio"])[0, 0].float().cpu().numpy()
        goal_visual[anchor] = _pool_visual(zg["visual"])[0, 0].float().cpu().numpy()
        goal_proprio[anchor] = _pool_proprio(zg["proprio"])[0, 0].float().cpu().numpy()
        true_visual[anchor] = _pool_visual(zt_visual).float().cpu().numpy()
        pred_visual[anchor] = _pool_visual(zp_visual).float().cpu().numpy()
        true_proprio[anchor] = _pool_proprio(zt_proprio).float().cpu().numpy()
        pred_proprio[anchor] = _pool_proprio(zp_proprio).float().cpu().numpy()
        default_true_score[anchor] = true_score.cpu().numpy()
        default_pred_score[anchor] = pred_score.cpu().numpy()
        latent_mse[anchor] = mse.cpu().numpy()
        print(f"Wall features {anchor + 1}/{anchors}", flush=True)

    np.savez_compressed(
        output_path,
        seeds=np.arange(seed_start, seed_start + anchors, dtype=np.int64),
        starts=starts,
        goals=goals,
        waypoint_y=waypoint_y,
        actions=actions,
        sampled_states=sampled_states,
        collisions=collisions,
        crossed_wall=crossed_wall,
        wall_x=wall_x_values,
        door_y=door_y_values,
        simulator_cost=simulator_cost,
        start_visual=start_visual,
        start_proprio=start_proprio,
        goal_visual=goal_visual,
        goal_proprio=goal_proprio,
        true_visual=true_visual,
        pred_visual=pred_visual,
        true_proprio=true_proprio,
        pred_proprio=pred_proprio,
        default_true_score=default_true_score,
        default_pred_score=default_pred_score,
        latent_mse=latent_mse,
    )
    total_candidates = anchors * candidate_count
    timing = {
        "anchors": anchors,
        "candidates": total_candidates,
        "simulator_seconds": simulator_seconds,
        "render_seconds": render_seconds,
        "encoder_seconds": encoder_seconds,
        "predictor_seconds": predictor_seconds,
        "simulator_candidates_per_second": total_candidates / simulator_seconds,
        "encoder_candidate_trajectories_per_second": total_candidates / encoder_seconds,
        "predictor_candidate_trajectories_per_second": total_candidates / predictor_seconds,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "output_bytes": output_path.stat().st_size,
    }
    output_path.with_suffix(".timing.json").write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    return timing


def goal_delta_features(
    visual: np.ndarray,
    proprio: np.ndarray,
    goal_visual: np.ndarray,
    goal_proprio: np.ndarray,
) -> np.ndarray:
    """Action-blind terminal/goal relation features for the small readout."""

    terminal = np.concatenate([visual[:, :, -1], proprio[:, :, -1]], axis=-1)
    goal = np.concatenate([goal_visual, goal_proprio], axis=-1)[:, None]
    delta = terminal - goal
    return np.concatenate([delta, np.abs(delta), np.square(delta)], axis=-1).astype(np.float32)

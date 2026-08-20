"""Deterministic cloned-state Wall rollouts and fixed candidate construction."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.transforms import Resize

from evals.simu_env_planning.envs.wall_env.envs.wall import DotWall
from evals.simu_env_planning.envs.wall_gym_wrap import DEFAULT_CFG


@dataclass(frozen=True)
class WallRollout:
    states: np.ndarray
    images: np.ndarray
    actions: np.ndarray
    collision_steps: np.ndarray


@dataclass(frozen=True)
class WallCandidateRollouts:
    """Candidate rollouts sampled at the model's latent interval."""

    sampled_states: np.ndarray
    actions: np.ndarray
    collision_steps: np.ndarray


def make_wall_env(*, wall_x: int = 32, door_y: int = 30, seed: int = 0) -> DotWall:
    config = deepcopy(DEFAULT_CFG)
    config.device = "cpu"
    config.fix_wall = True
    config.fix_wall_location = int(wall_x)
    config.fix_door_location = int(door_y)
    return DotWall(
        rng=np.random.default_rng(seed),
        wall_config=config,
        fix_wall=True,
        cross_wall=False,
        fix_wall_location=wall_x,
        fix_door_location=door_y,
        device="cpu",
    )


def _render_224(env: DotWall) -> np.ndarray:
    image = Resize((224, 224))(env.render()).permute(1, 2, 0)
    return image.cpu().numpy().astype(np.uint8)


def rollout_from_cloned_state(
    start: np.ndarray,
    actions: np.ndarray,
    *,
    wall_x: int = 32,
    door_y: int = 30,
    seed: int = 0,
) -> WallRollout:
    env = make_wall_env(wall_x=wall_x, door_y=door_y, seed=seed)
    start_tensor = torch.as_tensor(start, dtype=torch.float32)
    env.reset_to_state = start_tensor.clone()
    env.reset()
    states = [env.dot_position.detach().cpu().numpy().copy()]
    images = [_render_224(env)]
    collisions: list[bool] = []
    for action in np.asarray(actions, dtype=np.float32):
        before = env.dot_position.detach().cpu().numpy().copy()
        env.step(torch.as_tensor(action, dtype=torch.float32))
        after = env.dot_position.detach().cpu().numpy().copy()
        proposed = before + action * 2.0
        collisions.append(not np.allclose(after, proposed, atol=1e-5, rtol=0.0))
        states.append(after)
        images.append(_render_224(env))
    return WallRollout(
        states=np.asarray(states, dtype=np.float32),
        images=np.asarray(images, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.float32),
        collision_steps=np.asarray(collisions, dtype=bool),
    )


def waypoint_action_sequence(
    start: np.ndarray,
    goal: np.ndarray,
    waypoint_y: float,
    *,
    wall_x: int = 32,
    control_steps: int = 30,
) -> np.ndarray:
    """Create a deterministic two-segment open-loop route via the wall plane."""

    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    direction = 1.0 if goal[0] > start[0] else -1.0
    waypoint = np.asarray([wall_x + direction * 1.0, waypoint_y], dtype=np.float64)
    first_distance = float(np.linalg.norm(waypoint - start))
    second_distance = float(np.linalg.norm(goal - waypoint))
    # A unit action moves two pixels. Allocate enough controls to each segment,
    # then use all remaining controls on the second segment.
    first_steps = max(1, min(control_steps - 1, int(np.ceil(first_distance / 2.0))))
    second_steps = control_steps - first_steps
    actions = np.empty((control_steps, 2), dtype=np.float64)
    actions[:first_steps] = (waypoint - start) / (2.0 * first_steps)
    actions[first_steps:] = (goal - waypoint) / (2.0 * max(second_steps, 1))
    norms = np.linalg.norm(actions, axis=1, keepdims=True)
    actions = actions / np.maximum(norms, 1.0)
    return actions.astype(np.float32)


def waypoint_candidate_set(
    start: np.ndarray,
    goal: np.ndarray,
    *,
    wall_x: int = 32,
    candidate_count: int = 41,
    y_min: float = 8.0,
    y_max: float = 56.0,
    control_steps: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    waypoint_y = np.linspace(y_min, y_max, candidate_count, dtype=np.float32)
    actions = np.stack(
        [
            waypoint_action_sequence(
                start,
                goal,
                float(y),
                wall_x=wall_x,
                control_steps=control_steps,
            )
            for y in waypoint_y
        ],
        axis=0,
    )
    return waypoint_y, actions


def simulate_candidate_set(
    start: np.ndarray,
    actions: np.ndarray,
    *,
    wall_x: int = 32,
    door_y: int = 30,
    frameskip: int = 5,
    seed: int = 0,
) -> WallCandidateRollouts:
    """Run exact DotWall dynamics without paying to render every control step."""

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[-1] != 2:
        raise ValueError(f"Expected [candidate, control, 2], got {actions.shape}")
    if actions.shape[1] % frameskip:
        raise ValueError("Control count must be divisible by frameskip")
    env = make_wall_env(wall_x=wall_x, door_y=door_y, seed=seed)
    sampled = np.empty(
        (actions.shape[0], actions.shape[1] // frameskip + 1, 2), dtype=np.float32
    )
    collisions = np.zeros(actions.shape[:2], dtype=bool)
    for candidate, sequence in enumerate(actions):
        env.dot_position = torch.as_tensor(start, dtype=torch.float32).clone()
        sampled[candidate, 0] = env.dot_position.numpy()
        sample_index = 1
        for step, action in enumerate(sequence, start=1):
            before = env.dot_position.detach().cpu().numpy().copy()
            proposed = before + action * 2.0
            env.dot_position = env._calculate_next_position(torch.as_tensor(action))
            after = env.dot_position.detach().cpu().numpy().copy()
            collisions[candidate, step - 1] = not np.allclose(
                after, proposed, atol=1e-5, rtol=0.0
            )
            if step % frameskip == 0:
                sampled[candidate, sample_index] = after
                sample_index += 1
    return WallCandidateRollouts(sampled_states=sampled, actions=actions, collision_steps=collisions)


def render_states_224(
    states: np.ndarray,
    *,
    wall_x: int = 32,
    door_y: int = 30,
    chunk_size: int = 256,
) -> np.ndarray:
    """Render arbitrary Wall states exactly, batching the expensive resize."""

    states = np.asarray(states, dtype=np.float32)
    original_shape = states.shape[:-1]
    flat = torch.from_numpy(states.reshape(-1, 2))
    env = make_wall_env(wall_x=wall_x, door_y=door_y)
    wall = env._render_walls(env.wall_x, env.hole_y).bool()
    axis = torch.linspace(0, env.img_size - 1, steps=env.img_size)
    xx, yy = torch.meshgrid(axis, axis, indexing="xy")
    grid = torch.stack([xx, yy], dim=-1)
    resized: list[torch.Tensor] = []
    resize = Resize((224, 224))
    for start_index in range(0, len(flat), chunk_size):
        position = flat[start_index : start_index + chunk_size]
        squared_distance = (grid[None] - position[:, None, None]).pow(2).sum(dim=-1)
        dot = torch.exp(-squared_distance / (2 * env.dot_std * env.dot_std))
        dot = dot / dot.amax(dim=(1, 2), keepdim=True)
        image = torch.ones((len(position), 3, env.img_size, env.img_size), dtype=torch.uint8) * 255
        image[:, :, wall] = 0
        intensity = (dot * 255).to(torch.uint8)
        no_wall = ~wall
        image[:, 1, no_wall] = 255 - intensity[:, no_wall]
        image[:, 2, no_wall] = 255 - intensity[:, no_wall]
        resized.append(resize(image))
    output = torch.cat(resized).permute(0, 2, 3, 1).numpy()
    return output.reshape(*original_shape, 224, 224, 3)

"""Deterministic cloned-state Push-T rollouts and physical-event extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv


@dataclass
class PushTRollout:
    state0: np.ndarray
    actions: np.ndarray
    images: np.ndarray
    states: np.ndarray
    contacts: np.ndarray
    rewards: np.ndarray
    coverages: np.ndarray
    physical: dict[str, Any]
    task: dict[str, Any]


def state_to_proprio(states: np.ndarray) -> np.ndarray:
    """Extract the official [agent x, agent y, velocity x, velocity y] proprio."""

    states = np.asarray(states)
    return np.concatenate([states[..., :2], states[..., 5:7]], axis=-1).astype(np.float32)


def rollout_from_cloned_state(
    state: np.ndarray,
    actions: np.ndarray,
    *,
    seed: int,
    shape: str = "T",
    goal_pose: np.ndarray | None = None,
) -> PushTRollout:
    """Reset a fresh simulator to ``state`` and execute raw relative actions.

    Raw Push-T actions are in the official scaled convention: the simulator
    multiplies each 2-D action by 100 before applying its PD controller.
    """

    state = np.asarray(state, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if state.shape != (7,):
        raise ValueError(f"Expected seven-dimensional state, got {state.shape}")
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"Expected [T,2] actions, got {actions.shape}")

    env = PushTEnv(
        with_velocity=True,
        with_target=True,
        render_size=224,
        relative=True,
        action_scale=100,
        shape=shape,
    )
    try:
        env.seed(seed)
        env.reset_to_state = state.copy()
        obs0, reset_state = env.reset()
        if goal_pose is not None:
            env.set_task_goal(np.asarray(goal_pose, dtype=np.float64))
            obs0 = {**obs0, "visual": env.render(mode="rgb_array")}

        images = [obs0["visual"]]
        states = [reset_state]
        contacts: list[int] = []
        rewards: list[float] = []
        coverages: list[float] = []
        for action in actions:
            obs, reward, _, info = env.step(action)
            images.append(obs["visual"])
            states.append(info["state"])
            contacts.append(int(info["n_contacts"] > 0))
            rewards.append(float(reward))
            coverages.append(float(info["final_coverage"]))
    finally:
        env.close()

    contact_arr = np.asarray(contacts, dtype=np.int8)
    reward_arr = np.asarray(rewards, dtype=np.float32)
    coverage_arr = np.asarray(coverages, dtype=np.float32)
    any_contact = bool(contact_arr.any())
    physical = {
        "any_contact": any_contact,
        "contact_fraction": float(contact_arr.mean()) if len(contact_arr) else 0.0,
        "persistent_contact": bool(contact_arr.all()) if len(contact_arr) else False,
        "lost_contact": bool(contact_arr[:-1].any() and not contact_arr[-1]) if len(contact_arr) > 1 else False,
    }
    initial_coverage = float(coverage_arr[0]) if len(coverage_arr) else 0.0
    final_coverage = float(coverage_arr[-1]) if len(coverage_arr) else initial_coverage
    task = {
        "initial_coverage": initial_coverage,
        "final_coverage": final_coverage,
        "coverage_progress": final_coverage - initial_coverage,
        "success": bool(final_coverage >= 0.95),
        "cost": 1.0 - final_coverage,
    }
    return PushTRollout(
        state0=state.astype(np.float32),
        actions=actions.astype(np.float32),
        images=np.stack(images).astype(np.uint8),
        states=np.stack(states).astype(np.float32),
        contacts=contact_arr,
        rewards=reward_arr,
        coverages=coverage_arr,
        physical=physical,
        task=task,
    )


def constant_action_chunk(action: np.ndarray, control_steps: int) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (2,):
        raise ValueError(action.shape)
    return np.repeat(action[None], control_steps, axis=0)

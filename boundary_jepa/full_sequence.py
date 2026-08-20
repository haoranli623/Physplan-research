"""Auditable one-shot H6 Push-T planning with the official JEPA-WM CEM semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import time

import joblib
import numpy as np
import torch
from tensordict import TensorDict

from boundary_jepa.baseline import BaselineBundle
from boundary_jepa.features import _obs_tensordict, _pool_proprio, _pool_visual
from boundary_jepa.pusht import state_to_proprio
from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv


CANONICAL_GOAL_STATE = np.array(
    [256.0, 356.0, 256.0, 256.0, np.pi / 4.0, 0.0, 0.0], dtype=np.float64
)


@dataclass
class CEMTrace:
    candidates: np.ndarray  # [iterations, samples, horizon, action_dim], normalized
    scores: np.ndarray  # [iterations, samples]
    means_before: np.ndarray
    stds_before: np.ndarray
    means_after: np.ndarray
    stds_after: np.ndarray
    selected_mean: np.ndarray  # [horizon, action_dim]
    selected_score: float
    runtime_seconds: float

    @property
    def flat_candidates(self) -> np.ndarray:
        return self.candidates.reshape(-1, *self.candidates.shape[2:])


class TracedCEMPlanner:
    """Trace-preserving copy of the official diagonal-Gaussian CEM update.

    The sampling, mean-sample inclusion, elite selection, and mean/std updates
    match ``evals/.../planning/planning/planner.py::CEMPlanner``. The only new
    behavior is retaining candidates and distribution parameters.
    """

    def __init__(
        self,
        *,
        iterations: int,
        num_samples: int,
        num_elites: int,
        horizon: int,
        action_dim: int,
        var_scale: float,
        device: torch.device,
        seed: int,
        momentum_mean: float = 0.0,
        momentum_std: float = 0.0,
        max_norms: list[float] | None = None,
        max_norm_dims: list[list[int]] | None = None,
    ) -> None:
        self.iterations = int(iterations)
        self.num_samples = int(num_samples)
        self.num_elites = int(num_elites)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.var_scale = float(var_scale)
        self.device = device
        self.momentum_mean = float(momentum_mean)
        self.momentum_std = float(momentum_std)
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims or [[0, 1, 2], [6]]
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed))

    @torch.inference_mode()
    def plan(
        self,
        z_init: TensorDict,
        score_fn: Callable[[TensorDict, torch.Tensor], torch.Tensor],
    ) -> CEMTrace:
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = self.var_scale * torch.ones_like(mean)
        actions = torch.empty(
            self.horizon, self.num_samples, self.action_dim, device=self.device
        )
        candidates, scores = [], []
        means_before, stds_before, means_after, stds_after = [], [], [], []
        started = time.perf_counter()
        for _ in range(self.iterations):
            means_before.append(mean.float().cpu().numpy().copy())
            stds_before.append(std.float().cpu().numpy().copy())
            actions[:] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                self.horizon,
                self.num_samples,
                self.action_dim,
                device=self.device,
                generator=self.generator,
            )
            actions[:, 0, :] = mean
            if self.max_norms is not None:
                for h in range(self.horizon):
                    for dims, maxnorm in zip(self.max_norm_dims, self.max_norms):
                        actions[h, :, dims] = torch.clip(
                            actions[h, :, dims], min=-maxnorm, max=maxnorm
                        )
            cost = score_fn(z_init, actions).reshape(-1)
            if cost.shape[0] != self.num_samples or not torch.isfinite(cost).all():
                raise RuntimeError(f"Invalid CEM score tensor: {tuple(cost.shape)}")
            candidates.append(actions.transpose(0, 1).float().cpu().numpy().copy())
            scores.append(cost.float().cpu().numpy().copy())
            elite_idxs = torch.topk(-cost, self.num_elites, dim=0).indices
            elite_actions = actions[:, elite_idxs]
            new_mean = torch.mean(elite_actions, dim=1)
            new_std = torch.std(elite_actions, dim=1)
            mean = new_mean * (1.0 - self.momentum_mean) + mean * self.momentum_mean
            std = new_std * (1.0 - self.momentum_std) + std * self.momentum_std
            means_after.append(mean.float().cpu().numpy().copy())
            stds_after.append(std.float().cpu().numpy().copy())
        selected_score = float(score_fn(z_init, mean.unsqueeze(1)).reshape(-1)[0].cpu())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return CEMTrace(
            candidates=np.stack(candidates),
            scores=np.stack(scores),
            means_before=np.stack(means_before),
            stds_before=np.stack(stds_before),
            means_after=np.stack(means_after),
            stds_after=np.stack(stds_after),
            selected_mean=mean.float().cpu().numpy(),
            selected_score=selected_score,
            runtime_seconds=time.perf_counter() - started,
        )


def encode_single_state(bundle: BaselineBundle, state: np.ndarray) -> TensorDict:
    env = PushTEnv(
        with_velocity=True, with_target=True, render_size=224, relative=True,
        action_scale=100, shape="T", reset_to_state=np.asarray(state, dtype=np.float64),
    )
    try:
        env.seed(0)
        obs, reset_state = env.reset()
    finally:
        env.close()
    td = _obs_tensordict(obs["visual"][None, None], np.asarray(reset_state)[None, None])
    with torch.inference_mode():
        return bundle.model.encode(td.to(bundle.model.device))


def encode_goal(bundle: BaselineBundle, goal_state: np.ndarray = CANONICAL_GOAL_STATE) -> TensorDict:
    return encode_single_state(bundle, goal_state)


def pooled_terminal(pred: TensorDict) -> torch.Tensor:
    visual = _pool_visual(pred["visual"][-1:].transpose(0, 1))[:, 0]
    proprio = _pool_proprio(pred["proprio"][-1:].transpose(0, 1))[:, 0]
    return torch.cat([visual.float(), proprio.float()], dim=-1)


@dataclass(frozen=True)
class TorchReadout:
    mean: torch.Tensor
    scale: torch.Tensor
    coef: torch.Tensor
    intercept: torch.Tensor

    @classmethod
    def from_joblib(cls, path: str | Path, device: torch.device) -> "TorchReadout":
        pipeline = joblib.load(path)
        scaler = pipeline.named_steps["scaler"]
        ridge = pipeline.named_steps["ridge"]
        return cls(
            mean=torch.as_tensor(scaler.mean_, dtype=torch.float32, device=device),
            scale=torch.as_tensor(scaler.scale_, dtype=torch.float32, device=device),
            coef=torch.as_tensor(ridge.coef_, dtype=torch.float32, device=device),
            intercept=torch.as_tensor(ridge.intercept_, dtype=torch.float32, device=device),
        )

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return ((features - self.mean) / self.scale) @ self.coef + self.intercept


def make_model_scorers(
    bundle: BaselineBundle,
    z_goal: TensorDict,
    readout: TorchReadout | None = None,
    *,
    alpha: float = 0.1,
):
    model = bundle.model

    def predict(z_init: TensorDict, actions: torch.Tensor) -> TensorDict:
        return model.unroll(z_init, act_suffix=actions)

    def l2(z_init: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        pred = predict(z_init, actions)
        visual = (pred["visual"][-1] - z_goal["visual"][:, -1]).float().pow(2)
        proprio = (pred["proprio"][-1] - z_goal["proprio"][:, -1]).float().pow(2)
        return visual.mean(dim=tuple(range(1, visual.ndim))) + alpha * proprio.mean(
            dim=tuple(range(1, proprio.ndim))
        )

    def aligned(z_init: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        if readout is None:
            raise RuntimeError("No frozen readout supplied")
        return readout(pooled_terminal(predict(z_init, actions)))

    return l2, aligned, predict


class FeatureCapturingScorer:
    """Score predicted terminal latents while retaining their pooled features."""

    def __init__(
        self,
        bundle: BaselineBundle,
        z_goal: TensorDict,
        *,
        readout: TorchReadout | None = None,
        alpha: float = 0.1,
        model_query_batch_size: int | None = None,
    ) -> None:
        self.model = bundle.model
        self.z_goal = z_goal
        self.readout = readout
        self.alpha = float(alpha)
        self.model_query_batch_size = model_query_batch_size
        self.features: list[np.ndarray] = []
        self.l2_scores: list[np.ndarray] = []
        self.returned_scores: list[np.ndarray] = []

    @torch.inference_mode()
    def __call__(self, z_init: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        query_batch = self.model_query_batch_size or actions.shape[1]
        feature_chunks, l2_chunks = [], []
        for start in range(0, actions.shape[1], query_batch):
            pred = self.model.unroll(z_init, act_suffix=actions[:, start : start + query_batch])
            feature_chunks.append(pooled_terminal(pred))
            visual = (pred["visual"][-1] - self.z_goal["visual"][:, -1]).float().pow(2)
            proprio = (pred["proprio"][-1] - self.z_goal["proprio"][:, -1]).float().pow(2)
            l2_chunks.append(
                visual.mean(dim=tuple(range(1, visual.ndim)))
                + self.alpha * proprio.mean(dim=tuple(range(1, proprio.ndim)))
            )
        features = torch.cat(feature_chunks, dim=0)
        l2 = torch.cat(l2_chunks, dim=0)
        self.features.append(features.cpu().numpy())
        self.l2_scores.append(l2.cpu().numpy())
        returned = l2 if self.readout is None else self.readout(features)
        self.returned_scores.append(returned.cpu().numpy())
        return returned

    def stacked_features(self) -> np.ndarray:
        return np.concatenate(self.features, axis=0)

    def stacked_l2_scores(self) -> np.ndarray:
        return np.concatenate(self.l2_scores, axis=0)

    def stacked_returned_scores(self) -> np.ndarray:
        return np.concatenate(self.returned_scores, axis=0)


def sample_actionable_state(seed: int) -> np.ndarray:
    """Fresh near-contact state generator fixed independently of outcomes."""

    rng = np.random.default_rng(seed)
    for _ in range(10_000):
        block = rng.uniform([175.0, 175.0], [337.0, 337.0])
        angle = rng.uniform(-np.pi, np.pi)
        agent_angle = rng.uniform(-np.pi, np.pi)
        radius = rng.uniform(75.0, 125.0)
        agent = block + radius * np.array([np.cos(agent_angle), np.sin(agent_angle)])
        if np.all((agent >= 30.0) & (agent <= 482.0)):
            return np.array(
                [agent[0], agent[1], block[0], block[1], angle, 0.0, 0.0],
                dtype=np.float64,
            )
    raise RuntimeError("Could not sample an actionable Push-T state")


def denormalize_sequences(
    bundle: BaselineBundle, normalized_sequences: np.ndarray
) -> np.ndarray:
    values = np.asarray(normalized_sequences, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != bundle.model.action_dim:
        raise ValueError(values.shape)
    batch, horizon, flat_dim = values.shape
    if flat_dim % 2:
        raise ValueError("Model action dimension is not divisible by raw action dimension")
    raw_norm = values.reshape(batch, horizon * (flat_dim // 2), 2)
    with torch.no_grad():
        raw = bundle.preprocessor.denormalize_actions(torch.from_numpy(raw_norm))
    return raw.cpu().numpy().astype(np.float64)


@dataclass
class SimulationBatch:
    costs: np.ndarray
    final_states: np.ndarray
    final_images: np.ndarray | None
    contact_any: np.ndarray
    runtime_seconds: float


def simulate_sequences(
    state: np.ndarray,
    raw_sequences: np.ndarray,
    *,
    seed: int,
    goal_pose: np.ndarray = CANONICAL_GOAL_STATE[2:5],
    collect_final_images: bool = True,
) -> SimulationBatch:
    """Replay sequences from one cloned state while reusing one environment."""

    sequences = np.asarray(raw_sequences, dtype=np.float64)
    env = PushTEnv(
        with_velocity=True, with_target=True, render_size=224, relative=True,
        action_scale=100, shape="T",
    )
    n = len(sequences)
    costs = np.empty(n, dtype=np.float32)
    states = np.empty((n, 7), dtype=np.float32)
    images = np.empty((n, 224, 224, 3), dtype=np.uint8) if collect_final_images else None
    contacts = np.empty(n, dtype=np.int8)
    started = time.perf_counter()
    try:
        for i, sequence in enumerate(sequences):
            env.seed(seed)
            env.reset_to_state = np.asarray(state, dtype=np.float64).copy()
            original_render = env._render_frame
            env._render_frame = lambda mode: None
            any_contact = False
            try:
                env.reset()
                env.set_task_goal(np.asarray(goal_pose, dtype=np.float64))
                for action in sequence:
                    _, _, _, info = env.step(action)
                    any_contact |= bool(info["n_contacts"] > 0)
            finally:
                env._render_frame = original_render
            if images is not None:
                images[i] = env.render(mode="rgb_array")
            states[i] = info["state"]
            costs[i] = 1.0 - float(info["final_coverage"])
            contacts[i] = int(any_contact)
    finally:
        env.close()
    return SimulationBatch(
        costs=costs,
        final_states=states,
        final_images=images,
        contact_any=contacts,
        runtime_seconds=time.perf_counter() - started,
    )


def encode_endpoint_batch(
    bundle: BaselineBundle,
    images: np.ndarray,
    states: np.ndarray,
    *,
    batch_size: int = 128,
) -> np.ndarray:
    outputs = []
    device = bundle.model.device
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))
            td = _obs_tensordict(images[start:end, None], states[start:end, None]).to(device)
            encoded = bundle.model.encode(td)
            visual = _pool_visual(encoded["visual"])[:, 0]
            proprio = _pool_proprio(encoded["proprio"])[:, 0]
            outputs.append(torch.cat([visual.float(), proprio.float()], dim=-1).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def predict_trace_features(
    bundle: BaselineBundle,
    z_init: TensorDict,
    normalized_sequences: np.ndarray,
    *,
    batch_size: int = 300,
) -> np.ndarray:
    features = []
    model = bundle.model
    values = np.asarray(normalized_sequences, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            end = min(start + batch_size, len(values))
            actions = torch.from_numpy(values[start:end]).to(bundle.model.device).transpose(0, 1)
            pred = model.unroll(z_init, act_suffix=actions)
            features.append(pooled_terminal(pred).cpu().numpy())
    return np.concatenate(features, axis=0)

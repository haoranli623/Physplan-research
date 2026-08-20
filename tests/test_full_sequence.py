from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from boundary_jepa.full_sequence import TracedCEMPlanner
from evals.simu_env_planning.planning.planning.planner import CEMPlanner


def test_traced_cem_retains_official_mean_sample_and_update() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = TensorDict({"dummy": torch.zeros(1, 1, device=device)}, batch_size=[1, 1])

    def quadratic(_: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        return (actions - 0.25).pow(2).sum(dim=(0, 2))

    planner = TracedCEMPlanner(
        iterations=3,
        num_samples=16,
        num_elites=4,
        horizon=2,
        action_dim=3,
        var_scale=1.0,
        device=device,
        seed=17,
    )
    trace = planner.plan(z, quadratic)
    assert trace.candidates.shape == (3, 16, 2, 3)
    assert trace.scores.shape == (3, 16)
    np.testing.assert_allclose(trace.candidates[:, 0], trace.means_before, atol=0, rtol=0)
    np.testing.assert_allclose(trace.selected_mean, trace.means_after[-1], atol=0, rtol=0)
    for iteration in range(3):
        elite = np.argsort(trace.scores[iteration])[:4]
        np.testing.assert_allclose(
            trace.means_after[iteration], trace.candidates[iteration, elite].mean(axis=0),
            rtol=1e-5, atol=1e-6,
        )


def test_traced_cem_matches_official_implementation_on_cuda() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    z = torch.zeros(1, 1, device=device)

    def unroll(
        _: torch.Tensor,
        actions: torch.Tensor | None = None,
        act_suffix: torch.Tensor | None = None,
    ):
        return actions if actions is not None else act_suffix

    def objective(_: torch.Tensor, actions: torch.Tensor):
        return (actions - 0.25).pow(2).sum(dim=(0, 2))

    official_generator = torch.Generator(device=device).manual_seed(29)
    official = CEMPlanner(
        unroll=unroll, iterations=3, num_samples=16, num_elites=4,
        horizon=2, action_dim=3, var_scale=1.0,
        local_generator=official_generator, num_act_stepped=2,
    )
    official.set_objective(objective)
    official_result = official.plan(z)

    traced = TracedCEMPlanner(
        iterations=3, num_samples=16, num_elites=4, horizon=2,
        action_dim=3, var_scale=1.0, device=device, seed=29,
    ).plan(z, lambda _, actions: objective(actions, actions))
    np.testing.assert_allclose(
        traced.selected_mean, official_result.actions.cpu().numpy(), atol=0, rtol=0
    )
    np.testing.assert_allclose(
        traced.scores.min(axis=1), official_result.losses[:, 0].numpy(), atol=0, rtol=0
    )

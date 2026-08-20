from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from boundary_jepa.full_sequence import TracedCEMPlanner


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

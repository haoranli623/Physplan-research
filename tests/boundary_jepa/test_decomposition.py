import numpy as np

from boundary_jepa.decomposition import classify_bottleneck, discrete_cem_search, search_gaps


def test_discrete_cem_is_deterministic_and_selected_is_queried():
    scores = (np.linspace(0, 1, 21) - 0.72) ** 2
    left = discrete_cem_search(scores, iterations=4, samples=12, elites=3, seed=5)
    right = discrete_cem_search(scores, iterations=4, samples=12, elites=3, seed=5)
    assert np.array_equal(left.queried_indices, right.queried_indices)
    assert left.selected_index == right.selected_index
    assert left.queried_indices[-1] == left.selected_index
    assert abs(left.selected_index / 20 - 0.72) <= 0.1


def test_search_gaps_are_auditable_finite_set_quantities():
    costs = np.array([[0.5, 0.2, 0.0, 0.4]])
    result = discrete_cem_search(np.array([0.0, 0.1, 0.9, 1.0]), iterations=1, samples=2, elites=1, seed=1)
    gaps = search_gaps(costs, [result])
    assert gaps["g_search"][0] >= 0
    assert gaps["g_select"][0] >= 0
    assert np.isclose(gaps["total_search_regret"][0], gaps["g_search"][0] + gaps["g_select"][0])


def test_classification_prefers_material_dominant_metric_gap():
    summaries = {
        "representation_readout": {"rho_ci_low": 0.7, "regret_ci_high": 0.03},
        "prediction_gap": {"mean": 0.004, "ci_low": -0.002},
        "metric_gap": {"mean": 0.08, "ci_low": 0.06},
        "search_gap": {"mean": 0.02, "ci_low": 0.01},
        "selection_gap": {"mean": 0.07, "ci_low": 0.05},
    }
    assert classify_bottleneck(
        summaries,
        representation_rho_ci_low=0.5,
        representation_regret_ci_high=0.05,
        min_material_gap=0.02,
        dominance_ratio=1.5,
    ) == "DECISION METRIC"

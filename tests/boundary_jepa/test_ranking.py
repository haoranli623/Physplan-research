import numpy as np

from boundary_jepa.ranking import bootstrap_mean, endpoint_features, per_state_metrics


def test_endpoint_features_and_perfect_ranking_metrics():
    visual = np.arange(1 * 3 * 2 * 2, dtype=np.float32).reshape(1, 3, 2, 2)
    proprio = np.ones((1, 3, 2, 1), dtype=np.float32)
    features = endpoint_features(visual, proprio)
    assert features.shape == (1, 3, 3)
    costs = np.array([[0.1, 0.2, 0.4]])
    metrics = per_state_metrics(
        costs.copy(),
        costs,
        np.array([[1, 1, 0]]),
        np.array([1]),
        top_k=1,
        tie_tolerance=1e-4,
        boundary_near_cells=1,
        boundary_far_cells=1,
    )
    assert np.isclose(metrics["rho"][0], 1.0)
    assert np.isclose(metrics["pairwise_accuracy"][0], 1.0)
    assert metrics["selection_regret"][0] == 0.0
    assert metrics["boundary_pair_accuracy"][0] == 1.0


def test_bootstrap_mean_ignores_undefined_states():
    summary = bootstrap_mean(np.array([1.0, 3.0, np.nan]), samples=100, seed=7)
    assert summary["states"] == 2
    assert summary["mean"] == 2.0

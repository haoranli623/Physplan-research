import numpy as np

from boundary_jepa.pusht import rollout_from_cloned_state


def test_exact_cloned_replay_is_deterministic():
    state = np.array([256, 400, 256, 300, 0, 0, 0], dtype=np.float64)
    actions = np.repeat(np.array([[0.0, -0.5]], dtype=np.float64), 5, axis=0)
    first = rollout_from_cloned_state(state, actions, seed=17)
    second = rollout_from_cloned_state(state, actions, seed=17)
    np.testing.assert_allclose(first.states, second.states, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(first.images, second.images)
    np.testing.assert_array_equal(first.contacts, second.contacts)


def test_contact_and_task_labels_are_separate():
    state = np.array([256, 400, 256, 300, 0, 0, 0], dtype=np.float64)
    actions = np.repeat(np.array([[0.0, -0.5]], dtype=np.float64), 5, axis=0)
    rollout = rollout_from_cloned_state(state, actions, seed=17)
    assert set(rollout.physical) == {
        "any_contact",
        "contact_fraction",
        "persistent_contact",
        "lost_contact",
    }
    assert set(rollout.task) == {
        "initial_coverage",
        "final_coverage",
        "coverage_progress",
        "success",
        "cost",
    }

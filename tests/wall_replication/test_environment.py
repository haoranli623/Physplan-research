import numpy as np

from wall_replication.environment import (
    make_wall_env,
    render_states_224,
    rollout_from_cloned_state,
    simulate_candidate_set,
    waypoint_candidate_set,
)


def test_cloned_rollout_is_exact_across_rng_seeds():
    env = make_wall_env(seed=0)
    start, goal = env.generate_random_state(seed=123)
    _, actions = waypoint_candidate_set(start, goal, candidate_count=3)
    first = rollout_from_cloned_state(start, actions[1], seed=1)
    second = rollout_from_cloned_state(start, actions[1], seed=999)
    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_array_equal(first.images, second.images)


def test_fast_candidate_path_matches_direct_environment():
    env = make_wall_env(seed=0)
    start, goal = env.generate_random_state(seed=321)
    _, actions = waypoint_candidate_set(start, goal, candidate_count=3)
    batch = simulate_candidate_set(start, actions)
    direct = rollout_from_cloned_state(start, actions[1])
    np.testing.assert_array_equal(batch.sampled_states[1], direct.states[::5])
    rendered = render_states_224(batch.sampled_states[1])
    np.testing.assert_array_equal(rendered, direct.images[::5])


def test_waypoint_candidates_have_fixed_h6_geometry_and_cost_variation():
    env = make_wall_env(seed=0)
    start, goal = env.generate_random_state(seed=123)
    waypoint_y, actions = waypoint_candidate_set(start, goal, candidate_count=41)
    batch = simulate_candidate_set(start, actions)
    costs = np.linalg.norm(batch.sampled_states[:, -1] - goal, axis=1)
    assert waypoint_y.shape == (41,)
    assert actions.shape == (41, 30, 2)
    assert np.ptp(costs) > 4.0
    assert len(np.unique(np.round(costs, 3))) > 10

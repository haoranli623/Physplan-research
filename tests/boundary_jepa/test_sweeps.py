import numpy as np

from boundary_jepa.sweeps import SweepSpec, crossing_indices, sample_anchor_and_actions


def test_crossing_indices():
    np.testing.assert_array_equal(crossing_indices([1, 1, 0, 0]), [1])
    np.testing.assert_array_equal(crossing_indices([0, 1, 0]), [0, 1])


def test_action_slice_starts_toward_block():
    spec = SweepSpec(41, 1.2, 0.5, 85, 125, 5, 2)
    rng = np.random.default_rng(4)
    while True:
        try:
            state, actions, offsets, side = sample_anchor_and_actions(rng, spec)
            break
        except ValueError:
            pass
    toward = state[2:4] - state[:2]
    cosine = np.dot(toward, actions[0]) / (np.linalg.norm(toward) * np.linalg.norm(actions[0]))
    assert cosine > 0.999
    assert offsets[0] == 0
    assert np.sign(offsets[-1]) == side


def test_ranking_test_seed_is_disjoint():
    from boundary_jepa.sweeps import SPLIT_SEED_OFFSETS

    assert len(set(SPLIT_SEED_OFFSETS.values())) == len(SPLIT_SEED_OFFSETS)
    assert SPLIT_SEED_OFFSETS["ranking_test"] > SPLIT_SEED_OFFSETS["evaluation"]

import numpy as np

from boundary_jepa.analysis import interpolated_boundary, normalized_boundary_error


def test_interpolated_boundary_and_missing_penalty():
    offsets = np.array([0.0, 0.5, 1.0])
    boundary, count = interpolated_boundary(offsets, np.array([1.0, 0.75, 0.0]))
    assert count == 1
    assert np.isclose(boundary, 2.0 / 3.0)
    missing, count = interpolated_boundary(offsets, np.ones(3))
    assert count == 0 and np.isnan(missing)
    assert normalized_boundary_error(missing, boundary, offsets) == 1.0

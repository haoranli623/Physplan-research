import numpy as np

from boundary_jepa.probe import build_probe_features


def test_probe_features_never_require_actions():
    z0v = np.zeros((2, 3), dtype=np.float32)
    zfv = np.ones((2, 4, 2, 3), dtype=np.float32)
    z0p = np.zeros((2, 2), dtype=np.float32)
    zfp = np.ones((2, 4, 2, 2), dtype=np.float32)
    trajectory = build_probe_features(z0v, zfv, z0p, zfp, view="trajectory")
    endpoint = build_probe_features(z0v, zfv, z0p, zfp, view="endpoint")
    current = build_probe_features(z0v, zfv, z0p, zfp, view="current_only")
    assert trajectory.shape == (2, 4, 15)
    assert endpoint.shape == (2, 4, 10)
    assert current.shape == (2, 4, 5)

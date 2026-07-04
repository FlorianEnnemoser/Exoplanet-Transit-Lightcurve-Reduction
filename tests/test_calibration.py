# Master combine + per-method reduction arithmetic (S-7, S-8, S-25). GPL-3.0-or-later.
from __future__ import annotations

import numpy as np

from exotransit import calibration
from exotransit.calibration import MasterFrames


def test_build_master_mean_and_median():
    cube = np.array([[[1.0, 3.0]], [[3.0, 3.0]], [[5.0, 9.0]]])  # (3,1,2)
    assert np.allclose(calibration.build_master(cube, "mean"), [[3.0, 5.0]])
    assert np.allclose(calibration.build_master(cube, "median"), [[3.0, 3.0]])


def test_subtract_masters_per_method():
    frame = np.full((2, 2), 100.0)
    m = MasterFrames(dark=np.full((2, 2), 30.0), bias=np.full((2, 2), 10.0))
    assert np.allclose(calibration.subtract_masters(frame, m, "none"), 100.0)
    assert np.allclose(calibration.subtract_masters(frame, m, "standard"), 70.0)
    assert np.allclose(calibration.subtract_masters(frame, m, "bias"), 90.0)
    assert np.allclose(calibration.subtract_masters(frame, m, "dark_bias"), 60.0)

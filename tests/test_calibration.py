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


def test_build_master_flat_normalised_to_unit_median():
    # non-uniform flat with a known dark pedestal; result must have unit median (S-8).
    cube = np.array([[[10.0, 20.0], [30.0, 40.0]]] * 3) + np.arange(3)[:, None, None]
    dark = np.full((2, 2), 1.0)
    flat = calibration.build_master_flat(cube, dark)
    assert np.isclose(np.median(flat), 1.0)


def test_apply_flat_divides_and_is_identity_when_absent():
    frame = np.full((2, 2), 50.0)
    flat = np.array([[1.0, 2.0], [4.0, 5.0]])
    m = MasterFrames(flat=flat)
    assert np.allclose(calibration.apply_flat(frame, m), frame / flat)
    assert np.allclose(calibration.apply_flat(frame, MasterFrames()), frame)

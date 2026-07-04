# Manual-shift resolution (S-12, S-25). GPL-3.0-or-later.
from __future__ import annotations

from exotransit import tracking
from exotransit.config import ShiftSpec, Tracking


def test_manual_shift_recovered_after_trigger():
    tr = Tracking(
        mode="manual", reference_frame=0, manual_shifts=(ShiftSpec(2, 10, -5), ShiftSpec(4, 20, -8))
    )
    shifts = tracking.resolve_shifts(6, tr)
    # trigger at frame f affects frames f+1.. (legacy applies shift after frame f)
    assert shifts == [(0, 0), (0, 0), (0, 0), (10, -5), (10, -5), (20, -8)]


def test_off_mode_no_shift():
    tr = Tracking(mode="off", reference_frame=0, manual_shifts=())
    assert tracking.resolve_shifts(3, tr) == [(0, 0), (0, 0), (0, 0)]


def test_auto_shift_recovers_injected_offset():
    # S-25: phase cross-correlation recovers an injected drift to < 0.1 px.
    import numpy as np
    from scipy.ndimage import shift as ndshift

    yy, xx = np.mgrid[0:96, 0:96]
    ref = np.exp(-(((xx - 48) ** 2 + (yy - 50) ** 2) / 12.0))
    moved = ndshift(ref, (2.6, -1.3), order=3)  # (d_row=+2.6, d_col=-1.3)
    drow, dcol = tracking._measure_shift(ref, moved, upsample=20)
    assert abs(drow - 2.6) < 0.1
    assert abs(dcol - (-1.3)) < 0.1


def test_fixed_pattern_prefers_dark_over_bias():
    import numpy as np

    from exotransit.calibration import MasterFrames

    d, b = np.ones((4, 4)), np.zeros((4, 4))
    assert tracking._fixed_pattern(MasterFrames(dark=d, bias=b)) is d
    assert tracking._fixed_pattern(MasterFrames(dark=None, bias=b)) is b
    assert tracking._fixed_pattern(MasterFrames()) is None
    assert tracking._fixed_pattern(None) is None


def test_fixed_pattern_subtraction_unmasks_true_shift():
    # Regression (HAT-P-19 b): a saturated hot pixel pins raw correlation to zero;
    # subtracting the fixed pattern (master dark) recovers the real drift.
    import numpy as np
    from scipy.ndimage import shift as ndshift

    yy, xx = np.mgrid[0:96, 0:96]
    star = np.exp(-(((xx - 40) ** 2 + (yy - 44) ** 2) / 10.0)) * 500.0
    hot = np.zeros((96, 96))
    hot[70, 20] = 6e4  # fixed sensor defect, identical in both frames
    ref = star + hot
    moved = ndshift(star, (3.0, -2.0), order=3) + hot  # star drifts, hot pixel does not

    raw = tracking._measure_shift(ref, moved, upsample=20)
    assert abs(raw[0]) < 0.5 and abs(raw[1]) < 0.5  # hot pixel masks the drift
    fixed = tracking._measure_shift(ref - hot, moved - hot, upsample=20)
    assert abs(fixed[0] - 3.0) < 0.1 and abs(fixed[1] - (-2.0)) < 0.1  # drift recovered

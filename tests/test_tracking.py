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

# exotransit.timebase — light-curve time ordinate (S-14; ADR-0003).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""Time axis for the light curve.

P0 uses the raw ``TIME-OBS`` header string carried on each
:class:`~exotransit.io_fits.FrameMeta` (exactly the legacy behaviour, where the
header was used only as a plot label).

ponytail: proper BJD_TDB (mid-exposure, barycentric light-travel correction via
``astropy.time.Time`` + ``SkyCoord.light_travel_time``) is P1 — see S-14/ADR-0003.
It slots in by populating ``FrameMeta.bjd_tdb`` and switching ``labels`` below to
return BJD floats.
"""

from __future__ import annotations

from collections.abc import Iterable

from .io_fits import FrameMeta


def labels(frames: Iterable[FrameMeta]) -> list[str]:
    """Return the per-frame time labels used as the light-curve x-axis (P0)."""
    return [f.time_raw for f in frames]

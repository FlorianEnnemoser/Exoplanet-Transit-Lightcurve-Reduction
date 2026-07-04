# exotransit.tracking — drift registration (S-12; ADR-0004).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""Resolve a per-frame ``(dx, dy)`` crop-window offset series.

P0 supports manual shifts only — the legacy ``i``/``shift_x``/``shift_y``
triplets. A trigger at frame ``f`` sets an absolute offset ``(dx, dy)`` from the
initial coordinates that takes effect from frame ``f + 1`` onward and holds
until the next trigger (matching ``ExoplanetLightcurve.py:244``, where the shift
is applied *after* frame ``f`` is measured).

ponytail: automatic phase-cross-correlation registration is P1 (ADR-0004, S-12).
"""

from __future__ import annotations

from .config import Tracking


def resolve_shifts(n_frames: int, tracking: Tracking) -> list[tuple[int, int]]:
    """Return ``[(dx, dy), ...]`` of length ``n_frames``, one offset per frame."""
    if tracking.mode == "auto":
        raise NotImplementedError(
            "tracking.mode='auto' (phase cross-correlation) is P1; "
            "use mode='manual' or 'off' for now (ADR-0004)"
        )
    if tracking.mode == "off":
        return [(0, 0)] * n_frames

    # manual: last trigger with frame < j wins (legacy applies the shift after
    # measuring frame f, so it affects frames f+1 onward).
    triggers = sorted(tracking.manual_shifts, key=lambda s: s.frame)
    shifts: list[tuple[int, int]] = []
    for j in range(n_frames):
        dx = dy = 0
        for s in triggers:
            if s.frame < j:
                dx, dy = s.dx, s.dy
            else:
                break
        shifts.append((dx, dy))
    return shifts

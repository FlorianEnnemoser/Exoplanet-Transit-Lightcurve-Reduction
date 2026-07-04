# exotransit.tracking — drift registration (S-12; ADR-0004).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""Resolve a per-frame ``(dx, dy)`` crop-window offset series.

Two sources of offsets (ADR-0004, S-12):

* **manual** — the legacy ``i``/``shift_x``/``shift_y`` triplets. A trigger at
  frame ``f`` sets an absolute offset ``(dx, dy)`` from the initial coordinates
  that takes effect from frame ``f + 1`` onward and holds until the next trigger
  (matching ``ExoplanetLightcurve.py:244``, where the shift is applied *after*
  frame ``f`` is measured). ``off`` is the zero-shift case.
* **auto** — every light frame is registered against ``reference_frame`` by
  whole-frame phase cross-correlation (:func:`auto_shifts`); per-star sub-pixel
  centroiding then happens in photometry. Manual stays the explicit fallback.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from .calibration import MasterFrames
from .config import Tracking
from .io_fits import FrameSet


def resolve_shifts(n_frames: int, tracking: Tracking) -> list[tuple[int, int]]:
    """Return ``[(dx, dy), ...]`` of length ``n_frames`` for ``manual``/``off``.

    ``auto`` needs pixel data and is resolved by :func:`auto_shifts` instead.
    """
    if tracking.mode == "auto":
        raise ValueError(
            "tracking.mode='auto' needs pixel data; call auto_shifts(frames, tracking)"
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


def _measure_shift(
    reference: np.ndarray, image: np.ndarray, upsample: int = 10
) -> tuple[float, float]:
    """Sub-pixel displacement ``(d_row, d_col)`` of ``image`` relative to ``reference``.

    ``phase_cross_correlation`` returns the shift that registers ``image`` *onto*
    ``reference``; the star's displacement in ``image`` is its negation. Row is
    the crop ``dx`` axis, col the ``dy`` axis (photometry's legacy convention).
    """
    from skimage.registration import phase_cross_correlation

    shift = phase_cross_correlation(reference, image, upsample_factor=upsample)[0]
    return -float(shift[0]), -float(shift[1])


def _fixed_pattern(masters: MasterFrames | None) -> np.ndarray | None:
    """The frame's fixed sensor pattern to remove before correlating (dark > bias)."""
    if masters is None:
        return None
    return masters.dark if masters.dark is not None else masters.bias


def _prep(path, fixed: np.ndarray | None) -> np.ndarray:
    """Load a frame and strip the fixed pattern so real stars drive the correlation."""
    a = fits.getdata(path).astype(np.float32)
    return a - fixed if fixed is not None else a


def auto_shifts(
    frames: FrameSet, tracking: Tracking, masters: MasterFrames | None = None
) -> list[tuple[int, int]]:
    """Register every light frame against ``reference_frame`` by cross-correlation.

    The fixed sensor pattern (hot pixels / defective columns) is subtracted first
    via the master dark/bias — without it a saturated hot pixel dominates the
    correlation and pins every shift to zero (observed on HAT-P-19 b). Returns
    integer crop offsets; sub-pixel refinement is left to per-star centroiding in
    photometry (ADR-0004). ``reference_frame`` is clamped in range.
    """
    lights = frames.lights
    ref_idx = min(max(tracking.reference_frame, 0), len(lights) - 1)
    fixed = _fixed_pattern(masters)
    reference = _prep(lights[ref_idx].path, fixed)
    shifts: list[tuple[int, int]] = []
    for meta in lights:
        drow, dcol = _measure_shift(reference, _prep(meta.path, fixed))
        shifts.append((round(drow), round(dcol)))
    return shifts

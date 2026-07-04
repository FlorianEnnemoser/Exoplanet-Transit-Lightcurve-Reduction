# exotransit.lightcurve — differential light curve (S-15; ADR-0005).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""Form the differential light curve from science + calibrator fluxes.

P0 computes the per-calibrator diagnostic ratios ``F_sci / F_cal_k`` (the legacy
sci/cal1, sci/cal2 curves) and an equal-weight ensemble ``F_sci / mean(F_cal)``.
Flagged frames are ``NaN`` and propagate as ``NaN`` through the ratios.

ponytail: the Broeg-style variance-weighted ensemble with iterative rejection is
P1 (ADR-0005, S-15). For the acceptance invariant the pipeline derives the depth
from the first-calibrator ratio (the thesis' sci/cal1), not the ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .photometry import StarPhotometry


@dataclass(frozen=True)
class LightCurve:
    labels: list[str]  # per-frame time label (x-axis, P0 raw TIME-OBS)
    ratios: dict[str, np.ndarray]  # F_sci / F_cal_k, per calibrator name
    ensemble: np.ndarray  # F_sci / mean(F_cal), equal weight


def differential(
    sci: StarPhotometry, calibrators: list[StarPhotometry], labels: list[str]
) -> LightCurve:
    """Build per-calibrator ratios and the equal-weight ensemble curve."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = {c.star.name: sci.flux / c.flux for c in calibrators}
        cal_stack = np.vstack([c.flux for c in calibrators])
        ensemble = sci.flux / np.nanmean(cal_stack, axis=0)
    return LightCurve(labels=labels, ratios=ratios, ensemble=ensemble)

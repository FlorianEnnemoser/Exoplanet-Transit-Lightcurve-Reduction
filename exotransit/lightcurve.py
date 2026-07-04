# exotransit.lightcurve — differential light curve (S-15; ADR-0005).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""Form the differential light curve from science + calibrator fluxes.

Two products: the per-calibrator diagnostic ratios ``F_sci / F_cal_k`` (the
legacy sci/cal1, sci/cal2 curves) and a combined ensemble ``F_sci / C``.

The ensemble is a **Broeg-style variance-weighted artificial comparison** with
iterative rejection (S-15, ADR-0005) when N ≥ 2 calibrators *and* an
out-of-transit baseline window is supplied. Without a window (or with N = 1) it
degrades to the equal-weight ``F_sci / mean(F_cal)`` passthrough — this keeps the
acceptance-invariant (median-baseline) runs bit-for-bit unchanged, since those
call :func:`differential` without a window.

Flagged frames are ``NaN`` and propagate as ``NaN`` through the ratios; all
weight/rejection statistics use only finite baseline frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .photometry import StarPhotometry

_MAX_ITERS = 20
_EPS = 1e-12
_REJECT_FACTOR = 5.0  # drop a calibrator whose baseline scatter exceeds 5× the median


@dataclass(frozen=True)
class LightCurve:
    labels: list[str]  # per-frame time label (x-axis; raw TIME-OBS at P0)
    ratios: dict[str, np.ndarray]  # F_sci / F_cal_k, per calibrator name
    ensemble: np.ndarray  # F_sci / C (weighted) or F_sci / mean(F_cal) (equal)
    weights: dict[str, float] = field(default_factory=dict)  # normalised, used members only
    used: list[str] = field(default_factory=list)  # calibrators in the ensemble
    rejected: list[str] = field(default_factory=list)  # calibrators dropped by rejection


def differential(
    sci: StarPhotometry,
    calibrators: list[StarPhotometry],
    labels: list[str],
    *,
    bjd: np.ndarray | None = None,
    ingress: float | None = None,
    egress: float | None = None,
) -> LightCurve:
    """Build per-calibrator ratios and the ensemble curve.

    ``bjd``/``ingress``/``egress`` enable the Broeg-weighted ensemble; omit them
    (or pass a single calibrator) for the equal-weight passthrough.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = {c.star.name: sci.flux / c.flux for c in calibrators}

    names = [c.star.name for c in calibrators]
    base = _baseline_mask(bjd, ingress, egress, sci.flux)
    if len(calibrators) < 2 or base is None or np.count_nonzero(base) < 3:
        # equal-weight passthrough (invariant-safe path)
        with np.errstate(divide="ignore", invalid="ignore"):
            cal_stack = np.vstack([c.flux for c in calibrators])
            ensemble = sci.flux / np.nanmean(cal_stack, axis=0)
        w = {n: 1.0 / len(names) for n in names}
        return LightCurve(labels, ratios, ensemble, weights=w, used=list(names), rejected=[])

    ensemble, weights, used, rejected = _broeg(sci.flux, calibrators, names, base)
    return LightCurve(labels, ratios, ensemble, weights=weights, used=used, rejected=rejected)


def _baseline_mask(
    bjd: np.ndarray | None, ingress: float | None, egress: float | None, sci_flux: np.ndarray
) -> np.ndarray | None:
    """Out-of-transit, science-finite frames — the window all Broeg stats use."""
    if bjd is None or ingress is None or egress is None:
        return None
    bjd = np.asarray(bjd, dtype=float)
    return (np.isfinite(sci_flux)) & ((bjd < ingress) | (bjd > egress))


def _wmean(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-frame weighted mean over rows, ignoring NaN entries."""
    w = np.asarray(weights, dtype=float)[:, None]
    mask = np.isfinite(stack)
    num = np.nansum(np.where(mask, stack, 0.0) * w, axis=0)
    den = np.nansum(mask * w, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def _broeg(
    sci_flux: np.ndarray, calibrators: list[StarPhotometry], names: list[str], base: np.ndarray
) -> tuple[np.ndarray, dict[str, float], list[str], list[str]]:
    """Variance-weighted ensemble with iterative rejection (S-15).

    Each calibrator's baseline scatter is measured **against the ensemble of the
    others** (leave-one-out), so a single low-noise star cannot make the ensemble
    collapse onto itself. Weights are ``1/σ²``; a calibrator whose scatter is a
    gross outlier (> ``_REJECT_FACTOR`` × the median) is dropped, one per pass,
    until the set is stable or a single calibrator remains.
    """

    def _norm(flux: np.ndarray) -> np.ndarray:
        m = np.nanmedian(flux[base])
        return flux / m if m and np.isfinite(m) else flux

    sci_n = _norm(sci_flux)
    cals_n = {n: _norm(c.flux) for n, c in zip(names, calibrators)}

    def _comp(members: list[str], weights: dict[str, float]) -> np.ndarray:
        stack = np.vstack([cals_n[m] for m in members])
        return _wmean(stack, np.array([weights[m] for m in members]))

    used = list(names)
    # initial weights from each calibrator's own baseline scatter (S-15 step 1) so a
    # bad star is downweighted before it pollutes the leave-one-out comparisons.
    weights = {n: 1.0 / max(float(np.nanvar(cals_n[n][base])), _EPS) for n in names}
    sigma: dict[str, float] = {}
    for _ in range(_MAX_ITERS):
        for k in used:
            others = [m for m in used if m != k]
            if others:
                with np.errstate(divide="ignore", invalid="ignore"):
                    resid = (cals_n[k] / _comp(others, weights))[base]
                sigma[k] = float(np.nanstd(resid))
            else:
                sigma[k] = 1.0
        weights = {k: 1.0 / max(sigma[k] ** 2, _EPS) for k in used}
        if len(used) > 1:
            med = float(np.median([sigma[k] for k in used]))
            worst = max(used, key=lambda k: sigma[k])
            if med > 0.0 and sigma[worst] > _REJECT_FACTOR * med:
                used.remove(worst)
                continue  # re-weight the survivors before deciding again
        break

    with np.errstate(divide="ignore", invalid="ignore"):
        ensemble = sci_n / _comp(used, weights)
    total = sum(weights[n] for n in used) or 1.0
    norm_w = {n: weights[n] / total for n in used}
    rejected = [n for n in names if n not in used]
    return ensemble, norm_w, used, rejected

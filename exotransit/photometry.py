# exotransit.photometry — detection, aperture photometry, quality flags (S-10, S-11, S-13).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""Per-star, per-frame aperture photometry — the 2018-API port.

Modernises the legacy ``photutils``/``astropy`` calls (S-10: ``maxiters=``,
``.area`` property, ``photutils.detection``/``photutils.aperture`` imports,
``np.transpose([x, y])`` positions) while preserving the numerics that the
acceptance invariant pins.

**Legacy crop quirk preserved (deliberately):** the window is
``frame[x-h : x+h, y-h : y+h]`` — the star's *x* coordinate indexes the first
(row) axis. This is how the thesis picked coordinates in DS9; "fixing" the axis
order would move every star and break the invariant.

Detection guard (S-11/S-13): instead of the legacy silent ``sources[0]``, the
source nearest the window centre is chosen, and each frame carries a quality
flag; flagged frames get ``flux = NaN`` and are excluded downstream.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.detection import DAOStarFinder

from .calibration import MasterFrames, apply_cut, apply_flat, subtract_masters
from .config import Config, StarSpec
from .io_fits import FrameSet

logger = logging.getLogger(__name__)

_STAT_INDEX = {"mean": 0, "median": 1, "std": 2}  # sigma_clipped_stats return order


class QualityFlag:
    """Per-frame photometry outcome (names written verbatim to the CSV, S-21)."""

    OK = "OK"
    NO_SOURCE = "NO_SOURCE"
    OFF_CENTER = "OFF_CENTER"
    AMBIGUOUS = "AMBIGUOUS"
    SATURATED = "SATURATED"
    REGISTRATION_FAILED = "REGISTRATION_FAILED"


@dataclass(frozen=True)
class StarPhotometry:
    star: StarSpec
    flux: np.ndarray  # per-frame residual flux, NaN where flagged
    quality: list[str]  # per-frame QualityFlag


class ReductionError(Exception):
    """Raised when too many frames are flagged to trust the result (exit code 4)."""


def _measure_frame(window: np.ndarray, cfg: Config) -> tuple[float, str]:
    """Detect, guard, and photometer a single cropped window.

    Returns ``(flux, quality_flag)``. Flagged frames return ``(nan, flag)``.
    """
    ph = cfg.photometry
    center = cfg.crop_half_width  # window is ~(2h, 2h); centre ≈ (h, h)
    # OFF_CENTER: under P0 manual/coarse tracking the star drifts across the whole
    # crop between shift corrections, so it only needs to stay *inside* the window.
    # ponytail: auto tracking (P1, S-12) keeps it sub-pixel-centred → tighten this
    # back to ~2*fwhm per S-11 then.
    off_radius = float(cfg.crop_half_width)
    ambiguous_radius = ph.fwhm * 2.0

    stats = sigma_clipped_stats(window, sigma=ph.background_sigma, maxiters=ph.background_maxiters)
    stat = stats[_STAT_INDEX[ph.threshold_stat]]

    finder = DAOStarFinder(fwhm=ph.fwhm, threshold=ph.threshold_factor * stat)
    sources = finder(window)
    if sources is None or len(sources) == 0:
        return np.nan, QualityFlag.NO_SOURCE

    x = np.asarray(sources["xcentroid"])
    y = np.asarray(sources["ycentroid"])
    dist = np.hypot(x - center, y - center)
    near = int(np.argmin(dist))
    if dist[near] > off_radius:
        return np.nan, QualityFlag.OFF_CENTER
    if np.count_nonzero(dist <= ambiguous_radius) >= 2:
        return np.nan, QualityFlag.AMBIGUOUS

    cx, cy = float(x[near]), float(y[near])
    if _peak_near(window, cx, cy, ph.aperture_radius) >= ph.saturation:
        return np.nan, QualityFlag.SATURATED

    positions = np.transpose([[cx], [cy]])  # shape (1, 2), (x=col, y=row)
    aperture = CircularAperture(positions, r=ph.aperture_radius)
    annulus = CircularAnnulus(positions, r_in=ph.annulus_inner, r_out=ph.annulus_outer)
    phot = aperture_photometry(
        window, [aperture, annulus], method=ph.method, subpixels=ph.subpixels
    )
    bkg_mean = phot["aperture_sum_1"][0] / annulus.area
    flux = float(phot["aperture_sum_0"][0] - bkg_mean * aperture.area)
    return flux, QualityFlag.OK


def _peak_near(window: np.ndarray, cx: float, cy: float, radius: float) -> float:
    """Max pixel within a box of ``ceil(radius)`` around ``(cx, cy)`` (col, row)."""
    r = int(np.ceil(radius))
    r0 = max(int(round(cy)) - r, 0)
    c0 = max(int(round(cx)) - r, 0)
    box = window[r0 : int(round(cy)) + r + 1, c0 : int(round(cx)) + r + 1]
    return float(np.nanmax(box)) if box.size else -np.inf


def measure_star(
    frames: FrameSet,
    masters: MasterFrames,
    star: StarSpec,
    shifts: list[tuple[int, int]],
    cfg: Config,
) -> StarPhotometry:
    """Measure ``star`` across all light frames. Returns :class:`StarPhotometry`."""
    from astropy.io import fits

    h = cfg.crop_half_width
    flux = np.full(len(frames.lights), np.nan)
    quality: list[str] = []
    for meta in frames.lights:
        i = meta.index
        # S-8 order: subtract masters -> flat-field (full frame) -> crop -> cut.
        full = subtract_masters(
            fits.getdata(meta.path).astype(np.float32), masters, cfg.reduction.method
        )
        full = apply_flat(full, masters)
        dx, dy = shifts[i]
        # legacy indexing: x indexes rows (axis 0), y indexes cols (axis 1)
        x0, y0 = star.x + dx, star.y + dy
        crop = full[x0 - h : x0 + h, y0 - h : y0 + h]
        if crop.shape[0] < 2 * h or crop.shape[1] < 2 * h:
            quality.append(QualityFlag.REGISTRATION_FAILED)
            continue
        window = apply_cut(crop.astype(float), cfg)
        f, q = _measure_frame(window, cfg)
        flux[i] = f
        quality.append(q)
    n_flag = sum(1 for q in quality if q != QualityFlag.OK)
    logger.info("star %s: %d/%d frames flagged", star.name, n_flag, len(quality))
    return StarPhotometry(star=star, flux=flux, quality=quality)


def measure_all(
    frames: FrameSet, masters: MasterFrames, shifts: list[tuple[int, int]], cfg: Config
) -> list[StarPhotometry]:
    """Measure the science target and every calibrator; enforce the S-13 abort gate."""
    result = [measure_star(frames, masters, s, shifts, cfg) for s in cfg.stars]
    sci = result[0]
    n_flag = sum(1 for q in sci.quality if q != QualityFlag.OK)
    if n_flag > 0.5 * len(sci.quality):
        raise ReductionError(
            f"{n_flag}/{len(sci.quality)} science frames flagged (>50%); "
            f"aborting (exit 4). Check coordinates, tracking shifts, or focus."
        )
    return result

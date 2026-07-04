# exotransit.calibration — master frames + per-frame calibration (S-7, S-8).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""Build master dark/bias and apply per-method reduction.

Master combine honours ``master_dark_combine`` / ``master_bias_combine`` — the
legacy always-mean bias bug (``ExoplanetLightcurve.py:99``) is fixed here. Only
the masters the chosen ``reduction.method`` needs are built.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.stats import sigma_clip

from .config import Config
from .io_fits import FrameSet, load_cube

METHOD_NEEDS = {
    "none": (),
    "standard": ("dark",),
    "bias": ("bias",),
    "dark_bias": ("dark", "bias"),
}


@dataclass(frozen=True)
class MasterFrames:
    dark: np.ndarray | None = None
    bias: np.ndarray | None = None
    flat: np.ndarray | None = None  # ponytail: flat-fielding is P1 (R-14, S-8 step 2)


def build_master(cube: np.ndarray, combine: str) -> np.ndarray:
    """Combine a frame cube ``(n, rows, cols)`` along the stack axis."""
    if combine == "mean":
        return np.mean(cube, axis=0)
    if combine == "median":
        return np.median(cube, axis=0)
    raise ValueError(f"unknown combine mode: {combine!r}")


def build_masters(frames: FrameSet, cfg: Config) -> MasterFrames:
    """Build only the masters required by ``cfg.reduction.method``."""
    needs = METHOD_NEEDS[cfg.reduction.method]
    dark = (
        build_master(load_cube(frames.darks), cfg.reduction.master_dark_combine)
        if "dark" in needs
        else None
    )
    bias = (
        build_master(load_cube(frames.bias), cfg.reduction.master_bias_combine)
        if "bias" in needs
        else None
    )
    return MasterFrames(dark=dark, bias=bias)


def subtract_masters(frame: np.ndarray, masters: MasterFrames, method: str) -> np.ndarray:
    """Apply the reduction method's master subtraction to a full frame."""
    if method == "none":
        return frame
    if method == "standard":
        return frame - masters.dark
    if method == "bias":
        return frame - masters.bias
    if method == "dark_bias":
        return frame - masters.bias - masters.dark
    raise ValueError(f"unknown reduction method: {method!r}")


def apply_cut(window: np.ndarray, cfg: Config) -> np.ndarray:
    """Apply the optional per-frame pedestal cut to a cropped window (S-8 step 3)."""
    cut = cfg.reduction.cut
    if cut == "none":
        return window
    if cut == "median":
        return window - np.median(window)
    if cut == "average":
        return window - np.mean(window)
    if cut == "min":
        return window - np.min(window)
    if cut == "sigma_clip":
        # explicit mask policy (S-10): fill clipped pixels with NaN rather than
        # leaking a masked array downstream.
        clipped = sigma_clip(
            window, sigma=cfg.reduction.cut_sigma, maxiters=cfg.reduction.cut_maxiters
        )
        return clipped.filled(np.nan)
    raise ValueError(f"unknown cut mode: {cut!r}")

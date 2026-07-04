# exotransit.io_fits — FITS discovery, loading, header access, validation (S-9).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""Discover and validate the frame set for a run.

Frames ship flat in one directory with suffix naming (``*.BIAS.FIT`` = bias,
``*.DARK.FIT`` = dark, the rest = lights), so :func:`discover` classifies by
suffix and tolerates all three category paths pointing at the same directory.
Validation fails with a :class:`DataError` listing *every* offending file, not
just the first (R-18). Only ``paths.output`` is created on demand — input
directories are never auto-created (unlike the legacy ``os.makedirs``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from .config import Config

REQUIRED_BY_METHOD = {
    "none": ("lights",),
    "standard": ("lights", "darks"),
    "bias": ("lights", "bias"),
    "dark_bias": ("lights", "darks", "bias"),
}


class DataError(Exception):
    """Raised when input frames are missing, inconsistent, or malformed."""


@dataclass(frozen=True)
class FrameMeta:
    path: Path
    index: int
    time_raw: str
    exptime: float
    bjd_tdb: float | None = None  # ponytail: BJD_TDB is P1 (timebase.py, S-14)


@dataclass(frozen=True)
class FrameSet:
    lights: tuple[FrameMeta, ...]
    darks: tuple[Path, ...]
    bias: tuple[Path, ...]
    shape: tuple[int, int]


def _kind(path: Path) -> str:
    upper = path.name.upper()
    if ".BIAS." in upper:
        return "bias"
    if ".DARK." in upper:
        return "dark"
    return "light"


def _list_fits(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.upper() in (".FIT", ".FITS"))


def _dims(path: Path) -> tuple[int, int]:
    hdr = fits.getheader(path)
    return int(hdr["NAXIS2"]), int(hdr["NAXIS1"])  # (rows, cols)


def discover(cfg: Config) -> FrameSet:
    """Classify, validate, and order the frames referenced by ``cfg``.

    Raises :class:`DataError` accumulating all problems found.
    """
    errors: list[str] = []
    required = REQUIRED_BY_METHOD[cfg.reduction.method]

    all_light = [p for p in _list_fits(cfg.paths.lights) if _kind(p) == "light"]
    all_dark = [p for p in _list_fits(cfg.paths.darks) if _kind(p) == "dark"]
    all_bias = [p for p in _list_fits(cfg.paths.bias) if _kind(p) == "bias"]
    buckets = {"lights": all_light, "darks": all_dark, "bias": all_bias}
    dirs = {"lights": cfg.paths.lights, "darks": cfg.paths.darks, "bias": cfg.paths.bias}

    for cat in required:
        if not dirs[cat].is_dir():
            errors.append(f"[paths].{cat}: directory does not exist: {dirs[cat]}")
        elif not buckets[cat]:
            errors.append(f"[paths].{cat}: no {cat} FITS frames found in {dirs[cat]}")

    if errors:
        raise DataError(_format(errors))

    # dimensions — within category and across categories
    shape: tuple[int, int] | None = None
    for cat in required:
        cat_shape: tuple[int, int] | None = None
        for p in buckets[cat]:
            try:
                d = _dims(p)
            except Exception as exc:  # unreadable / not an image
                errors.append(f"{p}: cannot read FITS header ({exc})")
                continue
            if cat_shape is None:
                cat_shape = d
            elif d != cat_shape:
                errors.append(f"{p}: dimensions {d} differ from {cat} baseline {cat_shape}")
        if cat_shape is not None:
            if shape is None:
                shape = cat_shape
            elif cat_shape != shape:
                errors.append(f"[paths].{cat}: dimensions {cat_shape} differ from lights {shape}")

    # required headers on every light frame + collect timing
    metas: list[tuple[Path, str, float, str]] = []  # path, time_raw, exptime, sort_key
    for p in all_light:
        try:
            hdr = fits.getheader(p)
        except Exception as exc:
            errors.append(f"{p}: cannot read FITS header ({exc})")
            continue
        date_obs = hdr.get("DATE-OBS")
        time_obs = hdr.get("TIME-OBS")
        if date_obs is None and time_obs is None:
            errors.append(f"{p}: missing DATE-OBS/TIME-OBS header")
        if "EXPTIME" not in hdr:
            errors.append(f"{p}: missing EXPTIME header")
        sort_key = str(date_obs if date_obs is not None else time_obs)
        metas.append(
            (
                p,
                str(time_obs if time_obs is not None else date_obs),
                float(hdr.get("EXPTIME", 0.0)),
                sort_key,
            )
        )

    if errors:
        raise DataError(_format(errors))

    metas.sort(key=lambda m: m[3])  # deterministic order by observation time (R-18)
    lights = tuple(
        FrameMeta(path=p, index=i, time_raw=t, exptime=e) for i, (p, t, e, _) in enumerate(metas)
    )
    assert shape is not None
    return FrameSet(lights=lights, darks=tuple(all_dark), bias=tuple(all_bias), shape=shape)


def load_cube(paths: tuple[Path, ...] | list[Path]) -> np.ndarray:
    """Stack the given FITS frames into a float32 cube ``(n, rows, cols)``."""
    return np.stack([fits.getdata(p).astype(np.float32) for p in paths], axis=0)


def _format(errors: list[str]) -> str:
    return "input validation failed:\n  - " + "\n  - ".join(errors)

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
    time_raw: str  # raw header timestamp, kept for provenance (S-14)
    exptime: float
    bjd_tdb: float  # mid-exposure BJD_TDB — the light-curve ordinate (S-14)


@dataclass(frozen=True)
class FrameSet:
    lights: tuple[FrameMeta, ...]
    darks: tuple[Path, ...]
    bias: tuple[Path, ...]
    shape: tuple[int, int]
    flats: tuple[Path, ...] = ()  # optional; populated only when paths.flats is set (S-8)


def _kind(path: Path) -> str:
    upper = path.name.upper()
    if ".BIAS." in upper:
        return "bias"
    if ".DARK." in upper:
        return "dark"
    if ".FLAT." in upper:
        return "flat"
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
    # flats are optional (R-14, S-8): validated only when paths.flats is set.
    all_flat = (
        [p for p in _list_fits(cfg.paths.flats) if _kind(p) == "flat"] if cfg.paths.flats else []
    )
    buckets = {"lights": all_light, "darks": all_dark, "bias": all_bias, "flats": all_flat}
    dirs = {"lights": cfg.paths.lights, "darks": cfg.paths.darks, "bias": cfg.paths.bias}

    check = list(required)
    if cfg.paths.flats:  # set but empty/missing is an error; unset is fine
        dirs["flats"] = cfg.paths.flats
        check.append("flats")
    for cat in check:
        if not dirs[cat].is_dir():
            errors.append(f"[paths].{cat}: directory does not exist: {dirs[cat]}")
        elif not buckets[cat]:
            errors.append(f"[paths].{cat}: no {cat} FITS frames found in {dirs[cat]}")

    if errors:
        raise DataError(_format(errors))

    # dimensions — within category and across categories
    shape: tuple[int, int] | None = None
    for cat in check:
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
    # record: (path, date_obs, time_obs, exptime, time_raw)
    records: list[tuple[Path, str | None, str | None, float, str]] = []
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
        time_raw = str(time_obs if time_obs is not None else date_obs)
        records.append((p, date_obs, time_obs, float(hdr.get("EXPTIME", 0.0)), time_raw))

    if errors:
        raise DataError(_format(errors))

    # BJD_TDB per frame (S-14): target coord + observatory built once (S-2), reused.
    from . import timebase

    target = timebase.target_coord(cfg.system)
    site = timebase.observatory(cfg.system)
    metas: list[tuple[Path, str, float, float]] = []  # path, time_raw, exptime, bjd_tdb
    for p, date_obs, time_obs, exptime, time_raw in records:
        try:
            bjd = timebase.bjd_tdb(date_obs, time_obs, exptime, target, site)
        except Exception as exc:
            errors.append(f"{p}: cannot compute BJD_TDB from timestamp ({exc})")
            continue
        metas.append((p, time_raw, exptime, bjd))

    if errors:
        raise DataError(_format(errors))

    metas.sort(key=lambda m: m[3])  # deterministic order by BJD_TDB (R-18)
    lights = tuple(
        FrameMeta(path=p, index=i, time_raw=t, exptime=e, bjd_tdb=b)
        for i, (p, t, e, b) in enumerate(metas)
    )
    assert shape is not None
    return FrameSet(
        lights=lights,
        darks=tuple(all_dark),
        bias=tuple(all_bias),
        shape=shape,
        flats=tuple(all_flat),
    )


def load_cube(paths: tuple[Path, ...] | list[Path]) -> np.ndarray:
    """Stack the given FITS frames into a float32 cube ``(n, rows, cols)``."""
    return np.stack([fits.getdata(p).astype(np.float32) for p in paths], axis=0)


def _format(errors: list[str]) -> str:
    return "input validation failed:\n  - " + "\n  - ".join(errors)

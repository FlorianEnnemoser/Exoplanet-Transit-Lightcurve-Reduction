# exotransit.timebase — light-curve time ordinate (S-14; ADR-0003).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""BJD_TDB time base for the light curve.

Each frame's ordinate is the Barycentric Julian Date in the Barycentric Dynamical
Time scale (BJD_TDB) — the community standard for transit timing (ADR-0003),
comparable across observatories and epochs. It is computed from the FITS
``DATE-OBS``/``TIME-OBS`` and ``EXPTIME`` at **mid-exposure**, plus the target
ICRS coordinates and the observatory location, via ``astropy.time.Time`` and a
barycentric ``light_travel_time`` correction.

The raw header string is retained on :class:`~exotransit.io_fits.FrameMeta` for
provenance; ``bjd_tdb`` is the numeric ordinate used by plots, transit windows,
and fits.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from .config import System
from .io_fits import FrameMeta


def target_coord(system: System) -> SkyCoord:
    """Build the target ICRS :class:`~astropy.coordinates.SkyCoord` from config."""
    return SkyCoord(system.ra, system.dec, frame="icrs")


def observatory(system: System) -> EarthLocation:
    """Build the observatory :class:`~astropy.coordinates.EarthLocation` from config."""
    s = system.site
    return EarthLocation(lat=s.lat * u.deg, lon=s.lon * u.deg, height=s.height * u.m)


def _iso(date_obs: str | None, time_obs: str | None) -> str:
    """Assemble an ISO 8601 timestamp from the header date/time keywords.

    ``DATE-OBS`` usually already carries the full ``YYYY-MM-DDThh:mm:ss`` stamp
    (as in the WASP-52 b frames); when it holds only a date, the ``TIME-OBS``
    time part is joined onto it.
    """
    if date_obs and "T" in date_obs:
        return date_obs
    if date_obs and time_obs:
        return f"{date_obs}T{time_obs}"
    # last resort: whichever single field is present (Time will reject if unusable)
    return date_obs or time_obs or ""


def bjd_tdb(
    date_obs: str | None,
    time_obs: str | None,
    exptime: float,
    target: SkyCoord,
    site: EarthLocation,
) -> float:
    """Return the mid-exposure BJD_TDB for one frame.

    ``target`` and ``site`` are built once by the caller (S-2) and reused for
    every frame. Raises ``ValueError`` (via ``astropy``) if the timestamp will
    not parse — the caller turns that into a data-validation failure (R-18).
    """
    t_mid = Time(_iso(date_obs, time_obs), scale="utc", format="isot", location=site)
    t_mid = t_mid + (exptime / 2.0) * u.s
    ltt = t_mid.light_travel_time(target, kind="barycentric")
    return float((t_mid.tdb + ltt).jd)


def bjd_series(frames: Iterable[FrameMeta]) -> np.ndarray:
    """Return the per-frame BJD_TDB array — the light-curve x-axis (S-14)."""
    return np.array([f.bjd_tdb for f in frames], dtype=float)


def labels(frames: Iterable[FrameMeta]) -> list[str]:
    """Return the raw per-frame time strings (provenance / secondary axis label)."""
    return [f.time_raw for f in frames]

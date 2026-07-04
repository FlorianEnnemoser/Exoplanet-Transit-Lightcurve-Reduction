# exotransit.planet — transit depth + derived planet parameters (S-16, S-17).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
r"""Transit depth and derived planet parameters with Gaussian error propagation.

This is an **exact port** of the thesis arithmetic
(``ExoplanetLightcurve.py:386-463``); the numerics are pinned by the acceptance
invariant (WASP-52 b standard reduction → R_p ≈ 1.15 R_Jup, ρ ≈ 400 kg/m³,
i ≈ 87.3°), so the formulas are reproduced verbatim, not "cleaned up".

Depth (legacy ``baseline_fit = "median"``):
    baseline = mean( median(first 20), median(last 20) )     [out of transit]
    in_transit = median(middle 20)
    depth d = baseline − in_transit                          [in ratio units]

Derived quantities (SI, with R_* in R_sun, a in au, P in days, t_dur in minutes):
    Δm  = −2.5 · log10(in_transit / baseline)
    R_p = sqrt( d · (R_* · R_sun)² )                         [metres]
    ρ   = 3 · M_p / (4π R_p³)
    inclination:  cos i = sqrt( ((R_*·R_sun + R_p)² − (sin(π t_dur/P)·a·au)²)
                                 / (a·au·1e10) )
The inclination relation and its ``1e10`` scale factor come straight from the
thesis code (``:453``); documented here per R-25. ``baseline_fit = "linear"`` (the
airmass-slope fit) is P1 (S-16).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import System

# Physical constants (metres, kg, seconds) — module constants, not config (S-4).
R_JUP = 69_911_000.0  # Jupiter radius [m]
R_SUN = 695_508_000.0  # Solar radius [m]
AU = 149_597_870_700.0  # astronomical unit [m]
M_JUP = 1.898e27  # Jupiter mass [kg]


@dataclass(frozen=True)
class PlanetParams:
    depth: float
    delta_mag: float
    rp_m: float
    rp_rjup: float
    e_rp_rjup: float
    density: float
    e_density: float
    inclination_deg: float
    max_inclination_deg: float


def _window_medians(ratio: np.ndarray) -> tuple[float, float, float]:
    """Return (baseline, in_transit, out_of_transit_mean) legacy medians."""
    n = ratio.size
    first = np.nanmedian(ratio[:20])
    last = np.nanmedian(ratio[n - 20 :])
    mid = np.nanmedian(ratio[n // 2 - 10 : n // 2 + 10])
    baseline = (first + last) / 2.0
    return baseline, mid, baseline


def compute(ratio: np.ndarray, sysd: System) -> PlanetParams:
    """Derive planet parameters from a differential light-curve ratio (sci/cal)."""
    baseline, in_transit, _ = _window_medians(ratio)
    depth = baseline - in_transit
    delta_mag = -2.5 * np.log10(in_transit / baseline)

    rstar_m = sysd.r_star * R_SUN
    rp = np.sqrt(depth * rstar_m**2)
    e_rp = np.sqrt(depth) * (sysd.r_star_err * R_SUN)

    mp_kg = sysd.m_planet * M_JUP
    density = (3 * mp_kg) / (4 * np.pi * rp**3)
    # legacy error form (:450) reproduced verbatim, including its un-squared 2nd term
    e_density = np.sqrt(
        (abs((-9 * mp_kg) / (4 * np.pi * rp**4)) * e_rp) ** 2
        + abs(3 / (4 * np.pi * rp**3) * sysd.m_planet_err * M_JUP)
    )

    a_m = sysd.semi_major_axis * AU
    sin_term = (np.sin(np.pi * sysd.transit_duration * 60 / (sysd.period * 86400)) * a_m) ** 2
    rprs = (rstar_m + rp) ** 2
    inclination = np.degrees(np.arccos(np.sqrt((rprs - sin_term) / (a_m * 1e10))))
    max_rprs = (rp - e_rp + (sysd.r_star - sysd.r_star_err) * R_SUN) ** 2
    max_inc = np.degrees(np.arccos(np.sqrt((max_rprs - sin_term) / (a_m * 1e10))))

    return PlanetParams(
        depth=float(depth),
        delta_mag=float(delta_mag),
        rp_m=float(rp),
        rp_rjup=float(rp / R_JUP),
        e_rp_rjup=float(e_rp / R_JUP),
        density=float(density),
        e_density=float(e_density),
        inclination_deg=float(inclination),
        max_inclination_deg=float(max_inc),
    )

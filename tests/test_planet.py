# Derived planet parameters + error propagation (S-17, S-25). GPL-3.0-or-later.
from __future__ import annotations

import numpy as np
from conftest import write_toml

from exotransit import config, planet
from exotransit.planet import AU, M_JUP, R_JUP, R_SUN


def _ratio(n: int, depth: float) -> np.ndarray:
    r = np.ones(n)
    r[n // 2 - 10 : n // 2 + 10] = 1.0 - depth  # in-transit dip
    return r


def test_rp_and_error_vs_hand(tmp_path, wasp_config_text):
    sysd = config.load(write_toml(tmp_path, wasp_config_text)).system
    d = 0.02
    p = planet.compute(_ratio(60, d), sysd)

    assert np.isclose(p.depth, d)
    assert np.isclose(p.delta_mag, -2.5 * np.log10(1 - d))

    rp = np.sqrt(d * (sysd.r_star * R_SUN) ** 2)
    assert np.isclose(p.rp_rjup, rp / R_JUP)
    assert np.isclose(p.e_rp_rjup, np.sqrt(d) * sysd.r_star_err * R_SUN / R_JUP)

    density = 3 * sysd.m_planet * M_JUP / (4 * np.pi * rp**3)
    assert np.isclose(p.density, density)


def test_linear_baseline_recovers_depth_under_airmass_slope(tmp_path, wasp_config_text):
    sysd = config.load(write_toml(tmp_path, wasp_config_text)).system
    # 60 frames over 0.2 d; ingress/egress bracket the middle third.
    bjd = np.linspace(0.0, 0.2, 60)
    ingress, egress = 0.07, 0.13
    slope, intercept = 0.5, 1.0  # baseline trend a + b*t
    depth = 0.02
    ratio = intercept + slope * bjd
    core = (bjd >= ingress + 0.006) & (bjd <= egress - 0.006)
    ratio[core] *= 1.0 - depth  # multiplicative dip on top of the trend

    p = planet.compute(ratio, sysd, bjd=bjd, ingress=ingress, egress=egress, baseline_fit="linear")
    assert np.isclose(p.depth, depth, atol=2e-3)

    # median path is unaffected by the linear-only args
    assert planet.compute(_ratio(60, 0.02), sysd).depth > 0


def test_inclination_formula(tmp_path, wasp_config_text):
    sysd = config.load(write_toml(tmp_path, wasp_config_text)).system
    p = planet.compute(_ratio(60, 0.024), sysd)
    # independent recompute of the documented relation
    rp = np.sqrt(0.024 * (sysd.r_star * R_SUN) ** 2)
    a_m = sysd.semi_major_axis * AU
    sin_term = (np.sin(np.pi * sysd.transit_duration * 60 / (sysd.period * 86400)) * a_m) ** 2
    rprs = (sysd.r_star * R_SUN + rp) ** 2
    expected = np.degrees(np.arccos(np.sqrt((rprs - sin_term) / (a_m * 1e10))))
    assert np.isclose(p.inclination_deg, expected)
    assert 80.0 < p.inclination_deg < 90.0

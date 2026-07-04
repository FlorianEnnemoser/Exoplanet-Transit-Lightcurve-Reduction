# Photometry sky subtraction + detection-guard paths (S-11, S-13, S-25).
# GPL-3.0-or-later.
from __future__ import annotations

import numpy as np
from conftest import write_toml

from exotransit import config
from exotransit.photometry import QualityFlag, _measure_frame


def _star_window(cx, cy, amp=2000.0, sigma=2.0, bg=100.0, extra=None, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:50, 0:50]

    def blob(x0, y0, a):
        return a * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))

    win = bg + rng.normal(0, 1.0, (50, 50)) + blob(cx, cy, amp)
    if extra is not None:
        win += blob(extra[0], extra[1], amp)
    return win


def _cfg(tmp_path, wasp_config_text):
    return config.load(write_toml(tmp_path, wasp_config_text))  # crop_half_width = 25


def test_centered_star_ok_and_positive(tmp_path, wasp_config_text):
    cfg = _cfg(tmp_path, wasp_config_text)
    flux, flag = _measure_frame(_star_window(25, 25), cfg)
    assert flag == QualityFlag.OK
    assert flux > 0


def test_flux_invariant_to_sky_pedestal(tmp_path, wasp_config_text):
    # local sky subtraction must cancel a constant background (hand-reasoned check)
    cfg = _cfg(tmp_path, wasp_config_text)
    f_low, _ = _measure_frame(_star_window(25, 25, bg=100.0), cfg)
    f_high, _ = _measure_frame(_star_window(25, 25, bg=800.0), cfg)
    assert abs(f_high - f_low) / f_low < 0.01


def test_no_source_flagged(tmp_path, wasp_config_text):
    cfg = _cfg(tmp_path, wasp_config_text)
    rng = np.random.default_rng(0)
    _, flag = _measure_frame(100 + rng.normal(0, 1.0, (50, 50)), cfg)
    assert flag == QualityFlag.NO_SOURCE


def test_off_center_flagged(tmp_path, wasp_config_text):
    cfg = _cfg(tmp_path, wasp_config_text)
    _, flag = _measure_frame(_star_window(3, 3), cfg)  # ~31 px from centre > 25
    assert flag == QualityFlag.OFF_CENTER


def test_ambiguous_flagged(tmp_path, wasp_config_text):
    cfg = _cfg(tmp_path, wasp_config_text)
    # two sources straddling the centre, both within 2*fwhm (6 px)
    _, flag = _measure_frame(_star_window(21, 25, extra=(29, 25)), cfg)
    assert flag == QualityFlag.AMBIGUOUS

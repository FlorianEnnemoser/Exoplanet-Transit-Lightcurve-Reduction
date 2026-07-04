# Config load + validation rules (S-5, S-25). GPL-3.0-or-later.
from __future__ import annotations

import pytest
from conftest import write_toml

from exotransit import config
from exotransit.config import ConfigError


def test_valid_load(tmp_path, wasp_config_text):
    cfg = config.load(write_toml(tmp_path, wasp_config_text))
    assert cfg.target == "WASP-52 b"
    assert cfg.reduction.method == "standard"
    assert [s.name for s in cfg.stars] == ["WASP52", "Calibrator_1", "Calibrator_2"]
    assert len(cfg.tracking.manual_shifts) == 6


@pytest.mark.parametrize(
    "replace, field",
    [
        ("aperture_radius = 4.0", "aperture_radius = 0.0"),  # aperture > 0
        ("annulus_inner = 6.0", "annulus_inner = 9.0"),  # inner < outer
        ('method = "standard"', 'method = "bogus"'),  # method enum
        ("fwhm = 3.0", "fwmh = 3.0"),  # unknown key
        ("crop_half_width = 25", "crop_half_width = 3"),  # crop >= ceil(outer)
        ("background_maxiters = 5", "background_maxiters = 0"),  # >= 1
    ],
)
def test_validation_rejects(tmp_path, wasp_config_text, replace, field):
    text = wasp_config_text.replace(replace, field)
    with pytest.raises(ConfigError):
        config.load(write_toml(tmp_path, text))


def test_error_names_the_field(tmp_path, wasp_config_text):
    text = wasp_config_text.replace("aperture_radius = 4.0", "aperture_radius = 0.0")
    with pytest.raises(ConfigError, match="aperture_radius"):
        config.load(write_toml(tmp_path, text))


def test_empty_calibrators_rejected(tmp_path, wasp_config_text):
    text = wasp_config_text.replace(
        """calibrators = [
  { name = "Calibrator_1", x = 425, y = 210 },
  { name = "Calibrator_2", x = 240, y = 437 },
]""",
        "calibrators = []",
    )
    with pytest.raises(ConfigError, match="calibrator"):
        config.load(write_toml(tmp_path, text))

# webapp.server.configgen — wizard state -> validated pipeline TOML (W-11, W-NFR-3).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""Turn the wizard's config-shaped state into a pipeline TOML.

Validation is shared with the CLI by round-tripping through
``tomli_w.dumps`` -> :func:`exotransit.config.load` (S-29): the very loader
that drives ``exotransit reduce`` is the single source of truth, so a config
the web app exports is by construction one the pipeline accepts (W-NFR-3).
"""

from pathlib import Path
from typing import Any

import tomli_w

from exotransit.config import ConfigError, load

# Pre-filled wizard state (W-19): the S-4 defaults, mirrored from the
# config.load() fallbacks. Drift is caught by the round-trip validation —
# a stale key here fails load() with "unknown key" in the test suite.
DEFAULT_STATE: dict[str, Any] = {
    "schema_version": 1,
    "observation": {"target": "", "casename": "run"},
    "paths": {"lights": "", "darks": "", "bias": "", "flats": "", "output": ""},
    "stars": {"science": None, "calibrators": [], "crop_half_width": 25},
    "reduction": {
        "method": "standard",
        "cut": "none",
        "cut_sigma": 3.0,
        "cut_maxiters": 1,
        "master_dark_combine": "median",
        "master_bias_combine": "median",
    },
    "photometry": {
        "fwhm": 3.0,
        "threshold_factor": 20.0,
        "threshold_stat": "std",
        "background_sigma": 3.0,
        "background_maxiters": 5,
        "aperture_radius": 4.0,
        "annulus_inner": 6.0,
        "annulus_outer": 8.0,
        "method": "subpixel",
        "subpixels": 5,
        "saturation": 65535.0,
    },
    "tracking": {"mode": "auto", "reference_frame": 0, "manual_shifts": []},
    "transit": {"predicted_start": "", "predicted_end": "", "baseline_fit": "linear"},
    "system": {
        "r_star": None,
        "r_star_err": 0.0,
        "semi_major_axis": None,
        "period": None,
        "m_planet": None,
        "m_planet_err": 0.0,
        "transit_duration": None,
        "ra": "",
        "dec": "",
        "site": {"name": "", "lat": None, "lon": None, "height": None},
    },
    "output": {
        "write_csv": True,
        "figures": False,
        "colormap": "gray_r",
        "figsize": [15, 10],
    },
    "logging": {"level": "INFO", "file": "exotransit.log"},
}


def _prune(value: Any) -> Any:
    """Drop unset values (``None`` / empty strings / empty tables) recursively.

    Numbers and booleans pass through untouched — ``0`` and ``False`` are
    real values. What remains is exactly what ``tomli_w`` can serialize and
    what ``config.load`` treats as "field present".
    """
    if isinstance(value, dict):
        pruned = {k: _prune(v) for k, v in value.items()}
        return {k: v for k, v in pruned.items() if v is not None and v != {} and v != []}
    if isinstance(value, list):
        return [v for v in (_prune(v) for v in value) if v is not None]
    if value is None or (isinstance(value, str) and value == ""):
        return None
    return value


def to_toml(state: dict[str, Any]) -> str:
    """Serialize the wizard state to TOML text, omitting unset fields."""
    return tomli_w.dumps(_prune(state) or {})


def validate(state: dict[str, Any], workdir: Path) -> dict[str, Any]:
    """Round-trip ``state`` through the pipeline's own config loader (S-5).

    Writes ``workdir/config.toml`` (the session keeps the latest attempt) and
    returns ``{"valid", "error", "toml"}`` — ``error`` is the ConfigError
    message naming the offending field (W-9/W-16), never an exception.
    """
    text = to_toml(state)
    path = workdir / "config.toml"
    path.write_text(text, encoding="utf-8")
    try:
        load(path)
    except ConfigError as exc:
        return {"valid": False, "error": str(exc), "toml": None}
    return {"valid": True, "error": None, "toml": text}

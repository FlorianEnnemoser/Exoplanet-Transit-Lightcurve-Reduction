# exotransit.outputs — per-frame CSV + JSON sidecar (S-21, S-22; ADR-0006).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""Machine-readable run outputs: an extended per-frame CSV and a JSON sidecar.

Together they are sufficient to regenerate any figure (R-23) and make a run
self-describing via config hash + input-file provenance (R-24).
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .config import Config
from .io_fits import FrameSet
from .lightcurve import LightCurve
from .photometry import StarPhotometry
from .planet import PlanetParams


def config_sha256(cfg: Config) -> str:
    """SHA-256 of the config file that drove the run (provenance / R-24)."""
    return hashlib.sha256(Path(cfg.source_path).read_bytes()).hexdigest()


def write_csv(
    path: Path, frames: FrameSet, photometry: list[StarPhotometry], lc: LightCurve
) -> None:
    """Write one row per light frame with flux, quality flags, and ratios."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sci = photometry[0]
    cals = photometry[1:]
    header = ["frame_index", "file", "time_raw", "bjd_tdb", "exptime", "flux_sci", "quality_sci"]
    for c in cals:
        header += [f"flux_cal_{c.star.name}", f"quality_cal_{c.star.name}"]
    header += [f"ratio_{c.star.name}" for c in cals] + ["ratio_ensemble"]

    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for meta in frames.lights:
            i = meta.index
            row = [
                i,
                meta.path.name,
                meta.time_raw,
                f"{meta.bjd_tdb:.8f}",
                meta.exptime,
                _num(sci.flux[i]),
                sci.quality[i],
            ]
            for c in cals:
                row += [_num(c.flux[i]), c.quality[i]]
            row += [_num(lc.ratios[c.star.name][i]) for c in cals]
            row += [_num(lc.ensemble[i])]
            w.writerow(row)


def write_json(
    path: Path, cfg: Config, params: PlanetParams, frames: FrameSet, started: str, finished: str
) -> None:
    """Write the JSON sidecar (provenance, method, derived params)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": cfg.schema_version,
        "target": cfg.target,
        "provenance": {
            "software": "exotransit",
            "version": __version__,
            "config_path": str(cfg.source_path),
            "config_sha256": config_sha256(cfg),
            "input_files": {
                "lights": [m.path.name for m in frames.lights],
                "darks": [p.name for p in frames.darks],
                "bias": [p.name for p in frames.bias],
            },
            "run_started_utc": started,
            "run_finished_utc": finished,
        },
        "reduction": {
            "method": cfg.reduction.method,
            "cut": cfg.reduction.cut,
            "tracking_mode": cfg.tracking.mode,
            "baseline_fit": cfg.transit.baseline_fit,
        },
        "results": {
            "depth": params.depth,
            "delta_mag": params.delta_mag,
            "r_p_rjup": {"value": params.rp_rjup, "err_catalogue": params.e_rp_rjup},
            "density_kg_m3": {"value": params.density, "err_total": params.e_density},
            "inclination_deg": {"value": params.inclination_deg, "max": params.max_inclination_deg},
        },
    }
    path.write_text(json.dumps(doc, indent=2))


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(x: float) -> str | float:
    """Serialise NaN as empty string so the CSV round-trips cleanly."""
    return "" if isinstance(x, float) and np.isnan(x) else float(x)

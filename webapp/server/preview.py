# webapp.server.preview — background reduction preview job (W-12, W-22; ADR-0012).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""Run the reduction far enough to show a differential light curve.

A thread-backed, in-process job (ADR-0012) that mirrors the
:func:`exotransit.pipeline.run` chain **up to the light curve only** — no
``planet.compute``, no file outputs — so the wizard can preview the tracked
target and its photometry before committing to a full run. Progress is reported
per measured star; the job store is per-process and lost on restart, which is
fine for the local single-user tool.
"""

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from exotransit import calibration, io_fits, lightcurve, photometry, timebase, tracking
from exotransit.config import ConfigError, load

from . import configgen

logger = logging.getLogger(__name__)

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def _clean(arr: np.ndarray) -> list[float | None]:
    """List with non-finite entries as ``None`` so the JSON stays valid."""
    return [float(v) if np.isfinite(v) else None for v in np.asarray(arr, dtype=float)]


def status(sid: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(sid)
        return dict(job) if job else None


def start(sid: str, state: dict[str, Any], workdir: Path) -> dict[str, Any]:
    """Start (or report an already-running) preview job for ``sid``."""
    with _LOCK:
        job = _JOBS.get(sid)
        if job and job["status"] == "running":
            return dict(job)
        _JOBS[sid] = {
            "status": "running",
            "progress": 0.0,
            "stage": "starting",
            "result": None,
            "error": None,
        }
    threading.Thread(target=_run, args=(sid, state, workdir), daemon=True).start()
    return dict(_JOBS[sid])


def _set(sid: str, **fields: Any) -> None:
    with _LOCK:
        _JOBS[sid].update(fields)


def _run(sid: str, state: dict[str, Any], workdir: Path) -> None:
    try:
        _set(sid, stage="loading config", progress=0.02)
        path = workdir / "preview.toml"
        path.write_text(configgen.to_toml(state), encoding="utf-8")
        cfg = load(path)

        _set(sid, stage="discovering frames", progress=0.05)
        frames = io_fits.discover(cfg)

        _set(sid, stage="building masters", progress=0.12)
        masters = calibration.build_masters(frames, cfg)

        _set(sid, stage="tracking", progress=0.15)
        if cfg.tracking.mode == "auto":
            shifts = tracking.auto_shifts(frames, cfg.tracking, masters)
        else:
            shifts = tracking.resolve_shifts(len(frames.lights), cfg.tracking)

        # measure each star, reporting progress per star (0.15 -> 0.95). Reuses
        # measure_star; skips measure_all's >50%-flagged abort so a partial
        # preview still renders. ponytail: per-star granularity; thread a
        # per-frame callback into measure_star if finer progress is wanted.
        n = len(cfg.stars)
        phot = []
        for k, star in enumerate(cfg.stars):
            _set(sid, stage=f"measuring {star.name}", progress=0.15 + 0.80 * k / max(n, 1))
            phot.append(photometry.measure_star(frames, masters, star, shifts, cfg))

        _set(sid, stage="combining", progress=0.96)
        labels = timebase.labels(frames.lights)
        ingress = egress = None
        # branch mirrors pipeline.run (median = invariant path, linear = Broeg + windows).
        # ponytail: ~10 duplicated lines keep the invariant-pinned run() untouched.
        if cfg.transit.baseline_fit == "linear":
            date0 = str(fits.getheader(frames.lights[0].path).get("DATE-OBS", "")).split("T")[0]
            ingress, egress = timebase.transit_bounds_bjd(date0, cfg.transit, cfg.system)
            bjd = timebase.bjd_series(frames.lights)
            lc = lightcurve.differential(
                phot[0], phot[1:], labels, bjd=bjd, ingress=ingress, egress=egress
            )
        else:
            lc = lightcurve.differential(phot[0], phot[1:], labels)

        result = {
            "labels": labels,
            "ensemble": _clean(lc.ensemble),
            "ratios": {name: _clean(r) for name, r in lc.ratios.items()},
            "science": {"name": cfg.science.name, "x": cfg.science.x, "y": cfg.science.y},
            "quality": phot[0].quality,
            "shifts": [[int(dx), int(dy)] for dx, dy in shifts],
            "ingress": ingress,
            "egress": egress,
        }
        _set(sid, status="done", progress=1.0, stage="done", result=result)
    except ConfigError as exc:
        _set(sid, status="error", stage="config", error=str(exc))
    except (
        Exception
    ) as exc:  # any reduction failure surfaces as a preview error, never crashes the server
        logger.exception("preview job failed for %s", sid)
        _set(sid, status="error", stage="reduction", error=str(exc))

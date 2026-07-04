# exotransit.pipeline — orchestration (S-1).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""Wire the stages together: ``run(config) -> RunResult``.

This is the single execution entry point (with the CLI). Importing any module
does nothing; all I/O and computation start here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from . import calibration, io_fits, lightcurve, outputs, photometry, planet, timebase, tracking
from .config import Config
from .io_fits import FrameSet
from .lightcurve import LightCurve
from .photometry import StarPhotometry
from .planet import PlanetParams

logger = logging.getLogger("exotransit")


@dataclass(frozen=True)
class RunResult:
    frames: FrameSet
    photometry: list[StarPhotometry]
    lightcurve: LightCurve
    params: PlanetParams
    config: Config


def run(cfg: Config) -> RunResult:
    """Execute the full reduction for ``cfg`` and write outputs."""
    started = outputs.utcnow()
    logger.info("discovering frames for %s (method=%s)", cfg.target, cfg.reduction.method)
    frames = io_fits.discover(cfg)
    cfg.paths.output.mkdir(parents=True, exist_ok=True)  # only output is created

    logger.info("building masters")
    masters = calibration.build_masters(frames, cfg)
    if cfg.tracking.mode == "auto":
        logger.info(
            f"auto-tracking: phase cross-correlation vs frame {cfg.tracking.reference_frame:d}"
        )
        shifts = tracking.auto_shifts(frames, cfg.tracking, masters)
    else:
        shifts = tracking.resolve_shifts(len(frames.lights), cfg.tracking)

    logger.info("measuring %d stars over %d frames", len(cfg.stars), len(frames.lights))
    phot = photometry.measure_all(frames, masters, shifts, cfg)
    labels = timebase.labels(frames.lights)
    lc = lightcurve.differential(phot[0], phot[1:], labels)

    # depth from the first calibrator ratio (the thesis' sci/cal1) — pins the
    # acceptance invariant. ponytail: ensemble-based depth is P1 (S-15).
    depth_ratio = lc.ratios[cfg.calibrators[0].name]
    params = planet.compute(depth_ratio, cfg.system)

    if cfg.output.write_csv:
        outputs.write_csv(cfg.paths.output / f"{cfg.casename}_lightcurve.csv", frames, phot, lc)
    outputs.write_json(
        cfg.paths.output / f"{cfg.casename}_result.json",
        cfg,
        params,
        frames,
        started,
        outputs.utcnow(),
    )

    if cfg.output.figures:
        from . import plots

        try:
            plots.render(cfg, lc, cfg.paths.output)
        except Exception as exc:  # R-19: plotting never aborts a completed run
            logger.warning("figure generation failed (non-fatal): %s", exc)

    logger.info(
        "done: R_p=%.3f R_Jup, rho=%.0f kg/m3, i=%.2f deg",
        params.rp_rjup,
        params.density,
        params.inclination_deg,
    )
    return RunResult(frames=frames, photometry=phot, lightcurve=lc, params=params, config=cfg)

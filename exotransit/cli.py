# exotransit.cli — command-line interface (S-6).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""``exotransit reduce CONFIG.toml`` / ``exotransit validate CONFIG.toml``.

Exit codes (S-6): 0 success; 2 config/validation error; 3 data error; 4
reduction error (too many frames flagged); 1 unexpected exception.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys

from . import config as config_mod
from . import pipeline
from .config import ConfigError
from .io_fits import DataError
from .photometry import ReductionError

EXIT_OK, EXIT_UNEXPECTED, EXIT_CONFIG, EXIT_DATA, EXIT_REDUCTION = 0, 1, 2, 3, 4


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _apply_overrides(cfg, args):
    """Apply CLI overrides after validation; re-validate the changed values."""
    if args.reduction is not None:
        if args.reduction not in config_mod.REDUCTION_METHODS:
            raise ConfigError(
                f"--reduction: got {args.reduction!r}, "
                f"must be one of {list(config_mod.REDUCTION_METHODS)}"
            )
        cfg = dataclasses.replace(
            cfg, reduction=dataclasses.replace(cfg.reduction, method=args.reduction)
        )
    if args.aperture is not None:
        if args.aperture <= 0:
            raise ConfigError(f"--aperture: got {args.aperture}, must be > 0")
        cfg = dataclasses.replace(
            cfg, photometry=dataclasses.replace(cfg.photometry, aperture_radius=args.aperture)
        )
    if args.output_dir is not None:
        from pathlib import Path

        cfg = dataclasses.replace(
            cfg, paths=dataclasses.replace(cfg.paths, output=Path(args.output_dir))
        )
    if args.no_figures:
        cfg = dataclasses.replace(cfg, output=dataclasses.replace(cfg.output, figures=False))
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="exotransit")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("reduce", help="run the full reduction")
    r.add_argument("config")
    r.add_argument("--reduction", choices=config_mod.REDUCTION_METHODS)
    r.add_argument("--aperture", type=float)
    r.add_argument("--output-dir")
    r.add_argument("--log-level", default=None)
    r.add_argument("--no-figures", action="store_true")
    r.add_argument(
        "--dry-run", action="store_true", help="validate + discover + print the run plan, then exit"
    )

    v = sub.add_parser("validate", help="parse + validate the config only")
    v.add_argument("config")

    args = parser.parse_args(argv)

    try:
        cfg = config_mod.load(args.config)
        if args.command == "validate":
            _configure_logging("INFO")
            logging.getLogger("exotransit").info("config OK: %s", args.config)
            return EXIT_OK

        cfg = _apply_overrides(cfg, args)
        _configure_logging(args.log_level or cfg.log_level)

        if args.dry_run:
            from . import io_fits

            frames = io_fits.discover(cfg)
            print(
                f"target={cfg.target} method={cfg.reduction.method} "
                f"lights={len(frames.lights)} darks={len(frames.darks)} "
                f"bias={len(frames.bias)} stars={[s.name for s in cfg.stars]}"
            )
            return EXIT_OK

        result = pipeline.run(cfg)
        p = result.params
        print(f"R_p   = {p.rp_rjup:.4f} +/- {p.e_rp_rjup:.4f} R_Jup")
        print(f"rho   = {p.density:.1f} +/- {p.e_density:.1f} kg/m^3")
        print(f"i     = {p.inclination_deg:.3f} deg (max {p.max_inclination_deg:.3f})")
        print(f"depth = {p.depth:.5f}  delta_mag = {p.delta_mag:.5f}")
        return EXIT_OK

    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except DataError as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return EXIT_DATA
    except ReductionError as exc:
        print(f"reduction error: {exc}", file=sys.stderr)
        return EXIT_REDUCTION
    except Exception as exc:  # pragma: no cover - unexpected
        logging.getLogger("exotransit").exception("unexpected failure")
        print(f"unexpected error: {exc}", file=sys.stderr)
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    sys.exit(main())

# exotransit.plots — matplotlib figure generation (S-23).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""Figure generation for a run.

ponytail: figures are P1 (R-19, S-23). At P0 ``output.figures`` defaults to
false and the pipeline skips this module. When implemented, every figure MUST be
wrapped so a plotting exception logs a warning and never aborts a run after
fluxes are computed (R-19), and the x-axis uses BJD_TDB (S-14).
"""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .lightcurve import LightCurve


def render(cfg: Config, lc: LightCurve, output_dir: Path) -> None:
    """Placeholder — figure rendering lands in P1 (S-23)."""
    raise NotImplementedError("figure generation is P1 (S-23); set output.figures = false")

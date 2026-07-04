"""Acceptance regression test — the invariant gate (R-21, S-26; ADR-0008).

Runs the full standard (dark-subtraction) reduction of WASP-52 b and asserts the
derived parameters against tolerances **frozen from the first successful run of
the ported code** (2026-07-04, astropy 6.1.7 / photutils 2.0.2 / numpy 2.2.6):

    R_p = 1.2198 R_Jup    rho = 336.1 kg/m^3    i = 87.254 deg

Catalogue / thesis values are R_p ≈ 1.15 R_Jup, rho ≈ 400 kg/m^3, i ≈ 87.3°.
The **inclination is reproduced to 0.05°**, confirming the port is structurally
correct; R_p / rho sit ~6-16 % from catalogue, consistent with the thesis'
"first-order" caveat (CLAUDE.md §7) and the photutils-2.x vs 2018 subpixel
differences. The honest widening of the R_p/rho error bars to cover this gap is
the deferred statistical-depth-uncertainty work (R-27, P1).

The committed down-sampled fixture (S-26) is not yet in the repo: the three
targets span the frame and drift ~145 px, so shrinking to a committable size
needs binning (rescaling coords/shifts) or a licensing decision on committing
the GPL frames. Until then this test runs against a local ``_WASP52b/`` set and
skips when it is absent (e.g. in CI).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from exotransit import config, pipeline

DATA = Path(__file__).resolve().parents[1] / "_WASP52b"
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "wasp52b.toml"

# Frozen reference (see module docstring).
REF_RP_RJUP = 1.2198
REF_DENSITY = 336.1
REF_INCLINATION = 87.254


@pytest.mark.skipif(
    not (DATA.is_dir() and any(DATA.glob("*.FIT"))),
    reason="WASP-52 b frames not present (see S-26: committed fixture pending)",
)
def test_acceptance_wasp52b(tmp_path):
    cfg = config.load(CONFIG)
    cfg = dataclasses.replace(
        cfg,
        paths=dataclasses.replace(cfg.paths, output=tmp_path),
        output=dataclasses.replace(cfg.output, figures=False),
    )
    result = pipeline.run(cfg)
    p = result.params

    assert p.rp_rjup == pytest.approx(REF_RP_RJUP, abs=0.03)
    assert p.density == pytest.approx(REF_DENSITY, abs=10.0)
    assert p.inclination_deg == pytest.approx(REF_INCLINATION, abs=0.1)
    # the invariant that reproduces catalogue precisely
    assert p.inclination_deg == pytest.approx(87.3, abs=0.2)

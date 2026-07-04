# Differential light curve: N=1 passthrough + Broeg rejection (S-15, S-25). GPL-3.0-or-later.
from __future__ import annotations

import numpy as np

from exotransit.config import StarSpec
from exotransit.lightcurve import differential
from exotransit.photometry import QualityFlag, StarPhotometry


def _star(name: str, flux: np.ndarray) -> StarPhotometry:
    role = "science" if name == "sci" else "calibrator"
    return StarPhotometry(
        star=StarSpec(name, 0, 0, role),  # type: ignore[arg-type]
        flux=flux,
        quality=[QualityFlag.OK] * len(flux),
    )


def test_n1_passthrough_equals_sci_over_cal1():
    n = 40
    sci = _star("sci", np.full(n, 100.0))
    cal = _star("c1", np.full(n, 50.0))
    lc = differential(sci, [cal], [""] * n)
    assert np.allclose(lc.ensemble, 2.0)
    assert lc.used == ["c1"] and lc.rejected == []


def test_broeg_rejects_noisy_calibrator():
    rng = np.random.default_rng(0)
    n = 60
    bjd = np.linspace(0.0, 0.2, n)
    ingress, egress = 0.08, 0.12
    t = np.arange(n)
    sci = _star("sci", 100.0 + rng.normal(0, 0.1, n))
    steady1 = _star("steady1", 50.0 + rng.normal(0, 0.05, n))
    steady2 = _star("steady2", 70.0 + rng.normal(0, 0.05, n))
    # a calibrator with a large ramp + heavy noise → should be rejected
    noisy = _star("noisy", 60.0 + 5.0 * t + rng.normal(0, 5.0, n))

    lc = differential(
        sci, [steady1, steady2, noisy], [""] * n, bjd=bjd, ingress=ingress, egress=egress
    )
    assert "noisy" in lc.rejected
    assert "noisy" not in lc.used
    assert set(lc.used) <= {"steady1", "steady2"} and lc.used
    assert abs(sum(lc.weights.values()) - 1.0) < 1e-9

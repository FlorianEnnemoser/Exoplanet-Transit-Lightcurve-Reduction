# CSV/JSON round-trip + config hash stability (S-21, S-22, S-25). GPL-3.0-or-later.
from __future__ import annotations

import csv
import json

import numpy as np
from conftest import write_toml

from exotransit import config, outputs
from exotransit.io_fits import FrameMeta, FrameSet
from exotransit.lightcurve import LightCurve
from exotransit.photometry import QualityFlag, StarPhotometry
from exotransit.planet import compute


def _fixture(tmp_path, wasp_config_text):
    cfg = config.load(write_toml(tmp_path, wasp_config_text))
    lights = tuple(
        FrameMeta(
            path=tmp_path / f"f{i}.fit",
            index=i,
            time_raw=f"00:00:{i:02d}",
            exptime=10.0,
            bjd_tdb=2457662.5 + i * 0.001,
        )
        for i in range(3)
    )
    fs = FrameSet(lights=lights, darks=(), bias=(), shape=(50, 50))
    sci = StarPhotometry(
        cfg.science,
        np.array([10.0, np.nan, 12.0]),
        [QualityFlag.OK, QualityFlag.NO_SOURCE, QualityFlag.OK],
    )
    cal = StarPhotometry(
        cfg.calibrators[0],
        np.array([5.0, 5.0, 6.0]),
        [QualityFlag.OK, QualityFlag.OK, QualityFlag.OK],
    )
    lc = LightCurve(
        labels=["a", "b", "c"],
        ratios={cal.star.name: sci.flux / cal.flux},
        ensemble=sci.flux / cal.flux,
    )
    return cfg, fs, [sci, cal], lc


def test_csv_roundtrip(tmp_path, wasp_config_text):
    cfg, fs, phot, lc = _fixture(tmp_path, wasp_config_text)
    path = tmp_path / "out.csv"
    outputs.write_csv(path, fs, phot, lc)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 3
    assert rows[0]["quality_sci"] == "OK"
    assert rows[1]["quality_sci"] == "NO_SOURCE"
    assert rows[1]["flux_sci"] == ""  # NaN serialised as empty
    assert float(rows[2]["flux_sci"]) == 12.0
    assert float(rows[0]["bjd_tdb"]) == 2457662.5  # BJD_TDB column present (S-14/S-21)


def test_json_and_hash_stability(tmp_path, wasp_config_text):
    cfg, fs, phot, lc = _fixture(tmp_path, wasp_config_text)
    ratio = np.ones(60)
    ratio[20:40] = 0.98  # non-zero depth so derived params are finite
    params = compute(ratio, cfg.system)
    path = tmp_path / "r.json"
    outputs.write_json(path, cfg, params, fs, "t0", "t1")
    doc = json.loads(path.read_text())
    assert doc["target"] == "WASP-52 b"
    assert doc["reduction"]["method"] == "standard"
    assert doc["provenance"]["config_sha256"] == outputs.config_sha256(cfg)
    assert len(doc["provenance"]["config_sha256"]) == 64

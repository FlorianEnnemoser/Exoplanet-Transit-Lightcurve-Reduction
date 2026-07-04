# tests.test_webapp — web backend: session -> summary -> config -> export (S-30).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""End-to-end M1 contract: the exported TOML is accepted by ``config.load``
(W-NFR-3) and validation errors name the offending field (W-9/W-16)."""

import copy
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient

from exotransit.config import load
from webapp.server import configgen, sessions
from webapp.server.main import app


def _write_fits(path: Path, shape=(64, 64), time_obs="21:00:00") -> None:
    hdu = fits.PrimaryHDU(np.random.default_rng(0).normal(100.0, 5.0, shape).astype(np.float32))
    hdu.header["DATE-OBS"] = "2016-08-29"
    hdu.header["TIME-OBS"] = time_obs
    hdu.header["EXPTIME"] = 30.0
    hdu.writeto(path)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSIONS_ROOT", tmp_path / "sessions")
    return TestClient(app)


@pytest.fixture
def data_dir(tmp_path):
    # flat layout with suffix naming, like the real _WASP52b/ directory
    d = tmp_path / "data"
    d.mkdir()
    for i in range(3):
        _write_fits(d / f"L{i:04d}.FIT", time_obs=f"21:0{i}:00")
    for i in range(2):
        _write_fits(d / f"D{i}.DARK.FIT")
        _write_fits(d / f"B{i}.BIAS.FIT")
    return d


def _valid_state(data_dir: Path, output_dir: Path) -> dict:
    state = copy.deepcopy(configgen.DEFAULT_STATE)
    state["observation"] = {"target": "WASP-52 b", "casename": "WASP52"}
    state["paths"].update(
        {
            "lights": str(data_dir),
            "darks": str(data_dir),
            "bias": str(data_dir),
            "output": str(output_dir),
        }
    )
    state["stars"]["science"] = {"name": "WASP52", "x": 30, "y": 30}
    state["stars"]["calibrators"] = [{"name": "Calibrator_1", "x": 10, "y": 10}]
    state["stars"]["crop_half_width"] = 10
    state["system"].update(
        {
            "r_star": 0.79,
            "r_star_err": 0.02,
            "semi_major_axis": 0.0272,
            "period": 1.74978,
            "m_planet": 0.46,
            "m_planet_err": 0.02,
            "transit_duration": 110,
            "ra": "23h13m58.76s",
            "dec": "+08d45m40.6s",
        }
    )
    state["system"]["site"] = {
        "name": "Lustbuehel",
        "lat": 47.0678,
        "lon": 15.4936,
        "height": 480.0,
    }
    return state


def test_wizard_roundtrip(client, data_dir, tmp_path):
    sid = client.post("/api/sessions", json={"lights": str(data_dir)}).json()["id"]

    summary = client.get(f"/api/sessions/{sid}/summary").json()
    assert summary["lights"]["count"] == 3
    assert summary["lights"]["dims"] == [64, 64]
    assert summary["lights"]["problems"] == []
    assert summary["lights"]["time_obs_range"] == ["21:00:00", "21:02:00"]
    assert len(summary["lights"]["frames"]) == 3
    assert summary["darks"]["problems"] == ["no directory configured"]

    state = _valid_state(data_dir, tmp_path / "out")
    assert client.put(f"/api/sessions/{sid}/config", json=state).json() == {"ok": True}

    verdict = client.post(f"/api/sessions/{sid}/validate").json()
    assert verdict["valid"], verdict["error"]

    resp = client.get(f"/api/sessions/{sid}/config/export")
    assert resp.status_code == 200
    assert 'filename="WASP52.toml"' in resp.headers["content-disposition"]
    exported = tmp_path / "exported.toml"
    exported.write_text(resp.text, encoding="utf-8")
    cfg = load(exported)  # the CLI's own loader accepts the export (W-NFR-3)
    assert cfg.science.x == 30
    assert cfg.reduction.method == "standard"


def test_validate_names_offending_field(client, data_dir, tmp_path):
    sid = client.post("/api/sessions", json={}).json()["id"]
    state = _valid_state(data_dir, tmp_path / "out")
    state["photometry"]["annulus_inner"] = 9.0  # inner > outer (8.0)
    client.put(f"/api/sessions/{sid}/config", json=state)
    verdict = client.post(f"/api/sessions/{sid}/validate").json()
    assert not verdict["valid"]
    assert "annulus_inner" in verdict["error"]
    assert client.get(f"/api/sessions/{sid}/config/export").status_code == 422


def test_summary_flags_dimension_mismatch(client, data_dir):
    _write_fits(data_dir / "L9999.FIT", shape=(32, 32))
    sid = client.post("/api/sessions", json={"lights": str(data_dir)}).json()["id"]
    problems = client.get(f"/api/sessions/{sid}/summary").json()["lights"]["problems"]
    assert any("differ" in p for p in problems)


def test_frame_png_and_bounds(client, data_dir):
    sid = client.post("/api/sessions", json={"lights": str(data_dir)}).json()["id"]
    resp = client.get(f"/api/sessions/{sid}/frames/0/png?scale=zscale")
    assert resp.status_code == 200
    assert resp.content.startswith(b"\x89PNG")
    assert "max-age" in resp.headers.get("cache-control", "").lower()
    assert client.get(f"/api/sessions/{sid}/frames/99/png").status_code == 404
    assert client.get(f"/api/sessions/{sid}/frames/0/png?scale=bogus").status_code == 400


def test_unknown_session_is_404(client):
    assert client.get("/api/sessions/not-a-uuid/summary").status_code == 404

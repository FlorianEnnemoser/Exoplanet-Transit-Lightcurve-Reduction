# webapp.server.main — FastAPI app + S-30 endpoints (ADR-0009).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""HTTP surface of the config-generator milestone (M1).

Run from the repo root::

    uv run --group web uvicorn webapp.server.main:app --reload

Serves the built SPA from ``webapp/client/dist`` when present (single-command
local-first deployment, W-NFR-4); during development the Vite dev server
proxies ``/api`` here instead.
"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from exotransit import io_fits

from . import archive, configgen, growth, preview, rendering, sessions

app = FastAPI(title="exotransit web", version="0.1.0")
# local single-user tool (W-NFR-4); open CORS keeps the Vite dev server happy
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SessionPaths(BaseModel):
    lights: str = ""
    darks: str = ""
    bias: str = ""
    flats: str = ""
    output: str = ""


def _session(sid: str) -> dict[str, Any]:
    try:
        return sessions.get(sid)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None


@app.post("/api/sessions")
def create_session(paths: SessionPaths) -> dict[str, Any]:
    """Start a session from server-side directories (W-1); empty paths allowed."""
    return sessions.create(paths.model_dump())


@app.post("/api/sessions/{sid}/upload")
async def upload(sid: str, category: str, files: list[UploadFile]) -> dict[str, Any]:
    """Upload FITS frames for one category (W-1)."""
    _session(sid)
    if category not in sessions.CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown category {category!r}, must be one of {list(sessions.CATEGORIES)}",
        )
    return await sessions.store_uploads(sid, category, files)


@app.get("/api/sessions/{sid}/summary")
def summary(sid: str) -> dict[str, Any]:
    """Per-category frame summary + problem flags (W-2); feeds the timeline (W-14)."""
    p = _session(sid)["state"]["paths"]

    def _path(key: str) -> Path | None:
        return Path(p[key]) if p.get(key) else None

    return io_fits.summarize(_path("lights"), _path("darks"), _path("bias"), _path("flats"))


@app.get("/api/sessions/{sid}/frames/{index}/png")
def frame_png(sid: str, index: int, scale: str = "zscale") -> Response:
    """Rendered light frame for the viewer (W-4); display-resolution only."""
    if scale not in rendering.SCALES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown scale {scale!r}, must be one of {list(rendering.SCALES)}",
        )
    lights = sessions.light_frames(_session(sid)["state"])
    if not 0 <= index < len(lights):
        raise HTTPException(
            status_code=404,
            detail=f"frame index {index} out of range (0..{len(lights) - 1})",
        )
    # Cache in the browser to kill redundant re-fetch/re-render while scrubbing.
    # Bounded max-age (not immutable): URLs are keyed by index, and lights can be
    # re-uploaded to the same session, so an index may change content — 300s self-heals.
    return Response(
        rendering.render_png(str(lights[index]), scale),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.get("/api/sessions/{sid}/frames/{index}/growth")
def frame_growth(sid: str, index: int) -> dict[str, Any]:
    """Background-subtracted curve of growth per picked star for one frame (295)."""
    state = _session(sid)["state"]
    lights = sessions.light_frames(state)
    if not 0 <= index < len(lights):
        raise HTTPException(status_code=404, detail=f"frame index {index} out of range")
    stars = state["stars"]
    picks = ([stars["science"]] if stars["science"] else []) + stars["calibrators"]
    half = int(stars["crop_half_width"])
    curves = [growth.curve(str(lights[index]), s, state["photometry"], half) for s in picks]
    return {"stars": curves}


@app.get("/api/lookup")
def lookup(target: str) -> dict[str, Any]:
    """NASA Exoplanet Archive parameter lookup by planet name (293, W-10/W-20)."""
    return archive.lookup(target)


@app.post("/api/sessions/{sid}/preview")
def start_preview(sid: str) -> dict[str, Any]:
    """Start (or report) the background reduction-preview job (297, W-12)."""
    session = _session(sid)
    return preview.start(sid, session["state"], sessions.session_dir(sid))


@app.get("/api/sessions/{sid}/preview")
def get_preview(sid: str) -> dict[str, Any]:
    """Poll the preview job's status/progress/result (297)."""
    _session(sid)
    job = preview.status(sid)
    if job is None:
        return {"status": "idle", "progress": 0.0, "stage": "", "result": None, "error": None}
    return job


@app.get("/api/sessions/{sid}/config")
def get_config(sid: str) -> dict[str, Any]:
    return _session(sid)["state"]


@app.put("/api/sessions/{sid}/config")
def put_config(sid: str, state: dict[str, Any]) -> dict[str, Any]:
    _session(sid)
    sessions.save(sid, state)
    return {"ok": True}


@app.post("/api/sessions/{sid}/validate")
def validate(sid: str) -> dict[str, Any]:
    """Round-trip the state through exotransit.config.load (W-9/W-16, W-NFR-3)."""
    session = _session(sid)
    return configgen.validate(session["state"], sessions.session_dir(sid))


@app.get("/api/sessions/{sid}/config/export")
def export(sid: str) -> PlainTextResponse:
    """Download the validated pipeline TOML (W-11)."""
    session = _session(sid)
    result = configgen.validate(session["state"], sessions.session_dir(sid))
    if not result["valid"]:
        raise HTTPException(status_code=422, detail=result["error"])
    casename = session["state"].get("observation", {}).get("casename") or "exotransit"
    return PlainTextResponse(
        result["toml"],
        media_type="application/toml",
        headers={"Content-Disposition": f'attachment; filename="{casename}.toml"'},
    )


_DIST = Path(__file__).resolve().parents[1] / "client" / "dist"
if _DIST.is_dir():  # built SPA present -> serve it; mounted last so /api wins
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="client")

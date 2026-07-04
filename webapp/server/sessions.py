# webapp.server.sessions — disk-backed session store (W-1, W-3, W-18).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""One directory per session under ``_websessions/``.

``session.json`` holds the wizard state (config-shaped, see
:data:`configgen.DEFAULT_STATE`); uploaded frames land in
``uploads/<category>/``. JSON on disk makes sessions independent, isolated,
and resumable for free.
"""
# ponytail: JSON-file sessions, move to sqlite if concurrent sessions ever matter.

import copy
import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from exotransit.io_fits import list_lights

from .configgen import DEFAULT_STATE

SESSIONS_ROOT = Path("_websessions")
CATEGORIES = ("lights", "darks", "bias", "flats")


def _dir(sid: str) -> Path:
    # session ids are server-generated UUIDs; reject anything else so a
    # client-supplied id can never traverse outside SESSIONS_ROOT.
    try:
        uuid.UUID(sid)
    except ValueError:
        raise FileNotFoundError(f"invalid session id {sid!r}") from None
    return SESSIONS_ROOT / sid


def _write(d: Path, session: dict[str, Any]) -> None:
    (d / "session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")


def create(paths: dict[str, str]) -> dict[str, Any]:
    """Create a session, seeding the state paths from ``paths`` (W-1)."""
    sid = str(uuid.uuid4())
    d = SESSIONS_ROOT / sid
    d.mkdir(parents=True)
    state = copy.deepcopy(DEFAULT_STATE)
    state["paths"].update({k: v for k, v in paths.items() if v})
    if not state["paths"]["output"]:
        state["paths"]["output"] = str(d / "output")
    session = {
        "id": sid,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "state": state,
    }
    _write(d, session)
    return session


def get(sid: str) -> dict[str, Any]:
    f = _dir(sid) / "session.json"
    if not f.is_file():
        raise FileNotFoundError(f"unknown session {sid!r}")
    return json.loads(f.read_text(encoding="utf-8"))


def save(sid: str, state: dict[str, Any]) -> dict[str, Any]:
    session = get(sid)
    session["state"] = state
    _write(_dir(sid), session)
    return session


def session_dir(sid: str) -> Path:
    return _dir(sid)


def light_frames(state: dict[str, Any]) -> list[Path]:
    """Ordered light frames for the viewer/timeline (filename order)."""
    lights = state["paths"].get("lights") or ""
    return list_lights(Path(lights)) if lights else []


async def store_uploads(sid: str, category: str, files: list[UploadFile]) -> dict[str, Any]:
    """Store uploaded FITS frames and point the state's category path at them."""
    session = get(sid)
    dest = _dir(sid) / "uploads" / category
    dest.mkdir(parents=True, exist_ok=True)
    stored = 0
    for f in files:
        name = Path(f.filename or "").name  # strip any client-sent directory parts
        if not name:
            continue
        (dest / name).write_bytes(await f.read())
        stored += 1
    session["state"]["paths"][category] = str(dest)
    _write(_dir(sid), session)
    return {"stored": stored, "path": str(dest), "state": session["state"]}

# webapp.server.archive — NASA Exoplanet Archive parameter lookup (W-10, W-20; ADR-0013).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""Look up system parameters by planet name from the NASA Exoplanet Archive.

Queries the Archive's TAP sync service over stdlib ``urllib`` (no ``astroquery``
dependency, ADR-0013) and maps the ``pscomppars`` composite-parameter columns
onto the wizard's ``[system]`` fields. Lookup is best-effort: any network or
parse failure returns an empty mapping with a human note (W-20 — manual entry
always remains available), never an exception to the caller.
"""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
_COLUMNS = "st_rad,st_raderr1,pl_orbsmax,pl_orbper,pl_bmassj,pl_bmassjerr1,pl_trandur,rastr,decstr"
_TIMEOUT = 10.0  # seconds; the tool is interactive, don't hang the wizard


def _query(name: str) -> list[dict[str, object]]:
    """Run the TAP sync query for ``name``; return decoded rows (may be empty)."""
    # single-quote escaping for the ADQL string literal
    safe = name.replace("'", "''")
    adql = f"select {_COLUMNS} from pscomppars where pl_name='{safe}'"
    url = f"{_TAP}?{urllib.parse.urlencode({'query': adql, 'format': 'json'})}"
    with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:  # noqa: S310 (fixed https host)
        return json.loads(resp.read().decode("utf-8"))


def lookup(name: str) -> dict[str, object]:
    """Return ``{"found", "values", "note"}`` for planet ``name``.

    ``values`` maps directly onto ``state["system"]`` keys, omitting any column
    the Archive left null. Never raises: failures come back as ``found=False``
    with a note explaining why.
    """
    name = (name or "").strip()
    if not name:
        return {"found": False, "values": {}, "note": "Enter a target name first."}
    try:
        rows = _query(name)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        logger.warning("archive lookup failed for %r: %s", name, exc)
        return {
            "found": False,
            "values": {},
            "note": f"Lookup failed ({exc}). Enter values manually.",
        }

    if not rows:
        return {
            "found": False,
            "values": {},
            "note": f"No Archive entry for {name!r}. Check the name (e.g. 'WASP-52 b').",
        }

    row = rows[0]

    def num(col: str) -> float | None:
        v = row.get(col)
        return float(v) if isinstance(v, (int, float)) else None

    trandur_h = num("pl_trandur")  # Archive reports transit duration in hours
    raderr = num("st_raderr1")
    masserr = num("pl_bmassjerr1")
    values: dict[str, object] = {
        "r_star": num("st_rad"),
        "r_star_err": abs(raderr) if raderr is not None else None,
        "semi_major_axis": num("pl_orbsmax"),
        "period": num("pl_orbper"),
        "m_planet": num("pl_bmassj"),
        "m_planet_err": abs(masserr) if masserr is not None else None,
        "transit_duration": round(trandur_h * 60.0, 1) if trandur_h is not None else None,
        "ra": row.get("rastr") or None,
        "dec": row.get("decstr") or None,
    }
    values = {k: v for k, v in values.items() if v is not None}
    return {
        "found": True,
        "values": values,
        "note": f"Filled {len(values)} field(s) from the NASA Exoplanet Archive.",
    }

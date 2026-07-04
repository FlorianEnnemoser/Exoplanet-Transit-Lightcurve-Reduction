# exotransit.config — TOML load + validation (S-4, S-5; ADR-0001).
# Copyright (C) 2018-2026 Florian Ennemoser. GPL-3.0-or-later.
"""Load and validate a pipeline TOML config into frozen dataclasses.

The five mutually-exclusive legacy reduction flags collapse into
``reduction.method`` + ``reduction.cut`` so illegal combinations are
unrepresentable (R-8). ``load(path)`` fails fast with a :class:`ConfigError`
naming the offending field, its value, and the expectation (S-5).
"""

from __future__ import annotations

import difflib
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from astropy.coordinates import SkyCoord

REDUCTION_METHODS = ("none", "standard", "bias", "dark_bias")
CUT_MODES = ("none", "median", "average", "min", "sigma_clip")
COMBINE_MODES = ("mean", "median")
THRESHOLD_STATS = ("mean", "median", "std")
TRACKING_MODES = ("auto", "manual", "off")
BASELINE_FITS = ("median", "linear")


class ConfigError(Exception):
    """Raised when a config file is malformed or violates a validation rule."""


@dataclass(frozen=True)
class StarSpec:
    name: str
    x: int
    y: int
    role: Literal["science", "calibrator"]


@dataclass(frozen=True)
class ShiftSpec:
    frame: int
    dx: int
    dy: int


@dataclass(frozen=True)
class Paths:
    lights: Path
    darks: Path
    bias: Path
    flats: Path | None
    output: Path


@dataclass(frozen=True)
class Reduction:
    method: str
    cut: str
    cut_sigma: float
    cut_maxiters: int
    master_dark_combine: str
    master_bias_combine: str


@dataclass(frozen=True)
class Photometry:
    fwhm: float
    threshold_factor: float
    threshold_stat: str
    background_sigma: float
    background_maxiters: int
    aperture_radius: float
    annulus_inner: float
    annulus_outer: float
    method: str
    subpixels: int
    saturation: float


@dataclass(frozen=True)
class Tracking:
    mode: str
    reference_frame: int
    manual_shifts: tuple[ShiftSpec, ...]


@dataclass(frozen=True)
class Transit:
    predicted_start: str
    predicted_end: str
    baseline_fit: str


@dataclass(frozen=True)
class Site:
    """Observatory location — needed for the BJD_TDB light-travel correction (S-14)."""

    name: str
    lat: float
    lon: float
    height: float


@dataclass(frozen=True)
class System:
    r_star: float
    r_star_err: float
    semi_major_axis: float
    period: float
    m_planet: float
    m_planet_err: float
    transit_duration: float
    ra: str  # ICRS RA, e.g. "23h13m58.76s" — required for BJD_TDB (S-14)
    dec: str  # ICRS Dec, e.g. "+08d45m40.6s"
    site: Site


@dataclass(frozen=True)
class Output:
    write_csv: bool
    figures: bool
    colormap: str
    figsize: tuple[float, float]


@dataclass(frozen=True)
class Config:
    schema_version: int
    target: str
    casename: str
    paths: Paths
    science: StarSpec
    calibrators: tuple[StarSpec, ...]
    crop_half_width: int
    reduction: Reduction
    photometry: Photometry
    tracking: Tracking
    transit: Transit
    system: System
    output: Output
    log_level: str
    log_file: str
    source_path: Path = field(default=Path("."))

    @property
    def stars(self) -> tuple[StarSpec, ...]:
        return (self.science, *self.calibrators)


# --- validation helpers -----------------------------------------------------

_ALLOWED_KEYS: dict[str, set[str]] = {
    "": {
        "schema_version",
        "observation",
        "paths",
        "stars",
        "reduction",
        "photometry",
        "tracking",
        "transit",
        "system",
        "output",
        "logging",
    },
    "observation": {"target", "casename"},
    "paths": {"lights", "darks", "bias", "flats", "output"},
    "stars": {"science", "calibrators", "crop_half_width"},
    "reduction": {
        "method",
        "cut",
        "cut_sigma",
        "cut_maxiters",
        "master_dark_combine",
        "master_bias_combine",
    },
    "photometry": {
        "fwhm",
        "threshold_factor",
        "threshold_stat",
        "background_sigma",
        "background_maxiters",
        "aperture_radius",
        "annulus_inner",
        "annulus_outer",
        "method",
        "subpixels",
        "saturation",
    },
    "tracking": {"mode", "reference_frame", "manual_shifts"},
    "transit": {"predicted_start", "predicted_end", "baseline_fit"},
    "system": {
        "r_star",
        "r_star_err",
        "semi_major_axis",
        "period",
        "m_planet",
        "m_planet_err",
        "transit_duration",
        "ra",
        "dec",
        "site",
    },
    "system.site": {"name", "lat", "lon", "height"},
    "output": {
        "write_csv",
        "figures",
        "colormap",
        "figsize",
        "save_light_frames",
        "save_reduced_frames",
    },
    "logging": {"level", "file"},
}


def _reject_unknown(table: dict[str, Any], section: str) -> None:
    allowed = _ALLOWED_KEYS.get(section)
    if allowed is None:
        return
    for key in table:
        if key not in allowed:
            near = difflib.get_close_matches(key, allowed, n=1)
            hint = f" (did you mean '{near[0]}'?)" if near else ""
            loc = f"[{section}]" if section else "top level"
            raise ConfigError(f"{loc}: unknown key '{key}'{hint}")


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _enum(value: Any, allowed: tuple[str, ...], field_name: str) -> str:
    _need(value in allowed, f"{field_name}: got {value!r}, must be one of {list(allowed)}")
    return value


def _star(raw: dict[str, Any], role: str, field_name: str) -> StarSpec:
    _need(
        isinstance(raw, dict) and {"name", "x", "y"} <= set(raw),
        f"{field_name}: must be a table with name, x, y",
    )
    _need(
        raw["x"] >= 0 and raw["y"] >= 0,
        f"{field_name}: coordinates must be >= 0, got x={raw['x']}, y={raw['y']}",
    )
    return StarSpec(str(raw["name"]), int(raw["x"]), int(raw["y"]), role)  # type: ignore[arg-type]


def _site(raw: Any) -> Site:
    _need(
        isinstance(raw, dict) and {"lat", "lon", "height"} <= set(raw),
        "[system.site]: required table with lat, lon, height (needed for BJD_TDB)",
    )
    lat, lon, height = float(raw["lat"]), float(raw["lon"]), float(raw["height"])
    _need(-90.0 <= lat <= 90.0, f"[system.site].lat: got {lat}, must be in [-90, 90]")
    _need(-180.0 <= lon <= 360.0, f"[system.site].lon: got {lon}, must be in [-180, 360]")
    _need(math.isfinite(height), f"[system.site].height: got {height}, must be finite")
    return Site(name=str(raw.get("name", "")), lat=lat, lon=lon, height=height)


def load(path: str | Path) -> Config:
    """Parse and validate ``path``, returning a frozen :class:`Config`.

    Raises :class:`ConfigError` on any malformed field, naming it explicitly.
    """
    path = Path(path)
    _need(path.is_file(), f"config file not found: {path}")
    with path.open("rb") as fh:
        try:
            raw = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:  # pragma: no cover - passthrough
            raise ConfigError(f"{path}: invalid TOML — {exc}") from exc

    _reject_unknown(raw, "")
    for section in (
        "observation",
        "paths",
        "stars",
        "reduction",
        "photometry",
        "tracking",
        "transit",
        "system",
        "output",
        "logging",
    ):
        if section in raw:
            _reject_unknown(raw[section], section)
    if isinstance(raw.get("system"), dict) and "site" in raw["system"]:
        _reject_unknown(raw["system"]["site"], "system.site")

    obs = raw.get("observation", {})
    p = raw.get("paths", {})
    st = raw.get("stars", {})
    red = raw.get("reduction", {})
    ph = raw.get("photometry", {})
    tr = raw.get("tracking", {})
    tt = raw.get("transit", {})
    sysd = raw.get("system", {})
    out = raw.get("output", {})
    logd = raw.get("logging", {})

    # paths — relative to the current working directory (run from the repo root),
    # matching the legacy layout; absolute paths pass through unchanged.
    _need("output" in p, "[paths].output: required")
    paths = Paths(
        lights=Path(p.get("lights", "")),
        darks=Path(p.get("darks", "")),
        bias=Path(p.get("bias", "")),
        flats=Path(p["flats"]) if p.get("flats") else None,
        output=Path(p["output"]),
    )

    # stars
    _need("science" in st, "[stars].science: required")
    science = _star(st["science"], "science", "[stars].science")
    cal_raw = st.get("calibrators", [])
    _need(
        isinstance(cal_raw, list) and len(cal_raw) >= 1,
        "[stars].calibrators: at least one calibrator required (N >= 1)",
    )
    calibrators = tuple(
        _star(c, "calibrator", f"[stars].calibrators[{i}]") for i, c in enumerate(cal_raw)
    )
    crop = int(st.get("crop_half_width", 25))

    # reduction
    reduction = Reduction(
        method=_enum(red.get("method", "standard"), REDUCTION_METHODS, "[reduction].method"),
        cut=_enum(red.get("cut", "none"), CUT_MODES, "[reduction].cut"),
        cut_sigma=float(red.get("cut_sigma", 3.0)),
        cut_maxiters=int(red.get("cut_maxiters", 1)),
        master_dark_combine=_enum(
            red.get("master_dark_combine", "median"),
            COMBINE_MODES,
            "[reduction].master_dark_combine",
        ),
        master_bias_combine=_enum(
            red.get("master_bias_combine", "median"),
            COMBINE_MODES,
            "[reduction].master_bias_combine",
        ),
    )

    # photometry
    ai = float(ph.get("annulus_inner", 6.0))
    ao = float(ph.get("annulus_outer", 8.0))
    ar = float(ph.get("aperture_radius", 4.0))
    fwhm = float(ph.get("fwhm", 3.0))
    tf = float(ph.get("threshold_factor", 20.0))
    bmax = int(ph.get("background_maxiters", 5))
    _need(ar > 0, f"[photometry].aperture_radius: got {ar}, must be > 0")
    _need(
        0 < ai < ao, f"[photometry].annulus_inner: got {ai}, must satisfy 0 < inner < outer ({ao})"
    )
    _need(fwhm > 0, f"[photometry].fwhm: got {fwhm}, must be > 0")
    _need(tf > 0, f"[photometry].threshold_factor: got {tf}, must be > 0")
    _need(bmax >= 1, f"[photometry].background_maxiters: got {bmax}, must be >= 1")
    _need(
        crop >= math.ceil(ao),
        f"[stars].crop_half_width: got {crop}, must be >= ceil(annulus_outer) ({math.ceil(ao)})",
    )
    photometry = Photometry(
        fwhm=fwhm,
        threshold_factor=tf,
        threshold_stat=_enum(
            ph.get("threshold_stat", "std"), THRESHOLD_STATS, "[photometry].threshold_stat"
        ),
        background_sigma=float(ph.get("background_sigma", 3.0)),
        background_maxiters=bmax,
        aperture_radius=ar,
        annulus_inner=ai,
        annulus_outer=ao,
        method=str(ph.get("method", "subpixel")),
        subpixels=int(ph.get("subpixels", 5)),
        saturation=float(ph.get("saturation", 65535.0)),
    )

    # tracking
    shifts = tuple(
        ShiftSpec(int(s["frame"]), int(s["dx"]), int(s["dy"])) for s in tr.get("manual_shifts", [])
    )
    tracking = Tracking(
        mode=_enum(tr.get("mode", "manual"), TRACKING_MODES, "[tracking].mode"),
        reference_frame=int(tr.get("reference_frame", 0)),
        manual_shifts=shifts,
    )

    transit = Transit(
        predicted_start=str(tt.get("predicted_start", "")),
        predicted_end=str(tt.get("predicted_end", "")),
        baseline_fit=_enum(
            tt.get("baseline_fit", "median"), BASELINE_FITS, "[transit].baseline_fit"
        ),
    )

    for key in ("r_star", "semi_major_axis", "period", "m_planet", "transit_duration"):
        _need(key in sysd, f"[system].{key}: required")
    # ra/dec/site are required for BJD_TDB (S-14) and validated here (S-5).
    for key in ("ra", "dec"):
        _need(key in sysd, f"[system].{key}: required (needed for BJD_TDB)")
    try:
        SkyCoord(str(sysd["ra"]), str(sysd["dec"]), frame="icrs")
    except Exception as exc:
        raise ConfigError(
            f"[system].ra/dec: {sysd['ra']!r} / {sysd['dec']!r} do not parse as "
            f"ICRS coordinates ({exc})"
        ) from exc
    site = _site(sysd.get("site"))
    system = System(
        r_star=float(sysd["r_star"]),
        r_star_err=float(sysd.get("r_star_err", 0.0)),
        semi_major_axis=float(sysd["semi_major_axis"]),
        period=float(sysd["period"]),
        m_planet=float(sysd["m_planet"]),
        m_planet_err=float(sysd.get("m_planet_err", 0.0)),
        transit_duration=float(sysd["transit_duration"]),
        ra=str(sysd["ra"]),
        dec=str(sysd["dec"]),
        site=site,
    )

    fig = out.get("figsize", [15, 10])
    output = Output(
        write_csv=bool(out.get("write_csv", True)),
        figures=bool(out.get("figures", False)),  # ponytail: figures are P1 (S-23); default off
        colormap=str(out.get("colormap", "gray_r")),
        figsize=(float(fig[0]), float(fig[1])),
    )

    return Config(
        schema_version=int(raw.get("schema_version", 1)),
        target=str(obs.get("target", "unknown")),
        casename=str(obs.get("casename", "run")),
        paths=paths,
        science=science,
        calibrators=calibrators,
        crop_half_width=crop,
        reduction=reduction,
        photometry=photometry,
        tracking=tracking,
        transit=transit,
        system=system,
        output=output,
        log_level=str(logd.get("level", "INFO")).upper(),
        log_file=str(logd.get("file", "exotransit.log")),
        source_path=path,
    )

# Specifications

Engineering specifications for the refactor of the exoplanet transit lightcurve
reduction pipeline (Part I, deep detail) and the web data-input application
(Part II, milestone level). This document turns the *contract* in
`REQUIREMENTS.md` and the *decisions* in `ADR.md` into concrete, testable
specifications: module boundaries, schemas, algorithms, and acceptance criteria.

**How to read this document.**

- Every specification item carries a stable identifier `S-<n>` and a
  **Traces** line naming the requirement(s) (`R-*`, `W-*`, `NFR-*`) and
  decision(s) (`ADR-*`) it implements. The traceability matrix in §14 gives the
  reverse mapping.
- Keywords **MUST** / **SHOULD** / **MAY** follow RFC 2119, consistent with
  `REQUIREMENTS.md`.
- Items marked **`Status: proposed`** are *not yet* backed by an entry in
  `REQUIREMENTS.md`. They are suggestions surfaced during specification (§13).
  The maintainer promotes them into `REQUIREMENTS.md` (as new `R-*` entries) or
  drops them; until then they are non-binding.
- The **acceptance invariant** governs everything in Part I: the *standard*
  (dark-subtraction) reduction of WASP-52 b MUST keep reproducing
  R_p ≈ 1.15 R_Jup, ρ ≈ 400 kg/m³, i ≈ 87.3° within uncertainty (see §11.3).

---

# Part I — Pipeline refactor

## 1. Package layout & architecture

### S-1 Package structure

**Traces:** R-1, R-2, R-3, ADR-0002

The pipeline becomes an installable package `exotransit/`:

```
exotransit/
├── __init__.py        # version, public re-exports only — no I/O
├── config.py          # TOML load + validation, Config dataclasses (§2)
├── io_fits.py         # FITS discovery, loading, header access, validation
├── calibration.py     # master dark/bias/flat, frame calibration (§4)
├── tracking.py        # drift registration: auto + manual fallback (§5.3)
├── photometry.py      # detection, aperture photometry, quality flags (§5)
├── timebase.py        # BJD_TDB computation (§6)
├── lightcurve.py      # differential curve, ensemble comparison (§7)
├── planet.py          # transit depth, R_p, ρ, i + uncertainties (§8)
├── outputs.py         # CSV + JSON sidecar writers (§9)
├── plots.py           # all matplotlib figure generation (§10.2)
├── pipeline.py        # orchestration: run(config) -> RunResult
└── cli.py             # argument parsing, console entry point (§3)
```

- Importing **any** module MUST NOT read files, create directories, write
  output, or configure logging handlers (R-1). All execution starts from
  `pipeline.run()` or the CLI.
- `ExoplanetLightcurve.py` and `exo_input_values.py` remain in the repository
  untouched until the acceptance invariant is demonstrated by the new package
  (§11.3), then are removed in a dedicated commit.
- Licence stays **GPL v3**; every new module carries the licence header.

### S-2 Explicit state — no module globals

**Traces:** R-3

Hidden module-level state (`images_array`, `dark_master`, `bias_master`,
mutating crop coordinates) is replaced by values passed explicitly. Central
dataclasses (all `@dataclass(frozen=True)` unless noted):

```python
StarSpec(name: str, x: int, y: int, role: Literal["science", "calibrator"])
MasterFrames(dark: np.ndarray | None, bias: np.ndarray | None,
             flat: np.ndarray | None)                 # §4
FrameMeta(path: Path, index: int, time_raw: str, bjd_tdb: float,
          exptime: float)                              # §6
StarPhotometry(star: StarSpec, flux: np.ndarray,       # per-frame residual flux
               quality: np.ndarray)                    # per-frame QualityFlag, §5.4
RunResult(frames: list[FrameMeta], photometry: list[StarPhotometry],
          lightcurve: LightCurve, params: PlanetParams,  # §7, §8
          config: Config, provenance: Provenance)        # §9.3
```

The successor of `fluxtarget` takes everything it needs as parameters:

```python
def measure_star(frames: list[FrameMeta], masters: MasterFrames,
                 star: StarSpec, shifts: ShiftSeries,
                 cfg: PhotometryConfig) -> StarPhotometry: ...
```

### S-3 Packaging: uv + pyproject.toml

**Traces:** R-4, R-5, NFR-3, ADR-0002

- Single `pyproject.toml` with PEP 621 `[project]` metadata,
  `requires-python = ">=3.13"`, constrained runtime dependencies (`numpy`,
  `matplotlib`, `astropy`, `photutils`, `scikit-image` for ADR-0004,
  `tomli-w` for config writing per ADR-0001), and a committed `uv.lock`.
- Dev tools (`pytest`, `ruff`, `mypy`) live in a `[dependency-groups]` dev
  group.
- Console entry point via `[project.scripts]`:
  `exotransit = "exotransit.cli:main"`.
- `pip install .` MUST remain a working fallback (no uv-only mechanisms in the
  manifest).

## 2. Configuration

### S-4 TOML config schema

**Traces:** R-6, R-7, R-8, R-15, R-16, R-30, ADR-0001

One TOML file per target (R-7). The complete schema, shown with the WASP-52 b
values migrated from `exo_input_values.py`:

```toml
schema_version = 1              # R-30

[observation]
target = "WASP-52 b"
casename = "WASP52"             # tag appended to figure/CSV filenames

[paths]
lights = "_WASP52b/WASP52b"
darks  = "_WASP52b/Dark"
bias   = "_WASP52b/Bias"
flats  = ""                     # optional; empty = no flat-fielding (R-14)
output = "_WASP52b/images"

[stars]
science = { name = "WASP52", x = 595, y = 705 }
calibrators = [                 # N >= 1 entries (R-15)
  { name = "Calibrator_1", x = 425, y = 210 },
  { name = "Calibrator_2", x = 240, y = 437 },
]
crop_half_width = 25            # was pix_around_star

[reduction]
method = "standard"             # "none" | "standard" | "bias" | "dark_bias" (R-8)
cut = "none"                    # optional per-frame pedestal: "none" | "median" | "average" | "min" | "sigma_clip"
cut_sigma = 3.0                 # only read when cut = "sigma_clip"
cut_maxiters = 1
master_dark_combine = "median"  # "mean" | "median"
master_bias_combine = "median"  # honoured — fixes the current always-mean bug (§4.1)

[photometry]
fwhm = 3.0
threshold_factor = 20.0
threshold_stat = "std"          # "mean" | "median" | "std"  (was threshold[1] index)
background_sigma = 3.0
background_maxiters = 5
aperture_radius = 4.0
annulus_inner = 6.0
annulus_outer = 8.0
method = "subpixel"
subpixels = 5

[tracking]                      # ADR-0004
mode = "auto"                   # "auto" (phase cross-correlation) | "manual" | "off"
reference_frame = 0             # index of the registration reference
# manual fallback — the legacy i/shift_x/shift_y triplets:
manual_shifts = [
  { frame = 50,  dx = 30,  dy = -5  },
  { frame = 80,  dx = 55,  dy = -15 },
  { frame = 105, dx = 80,  dy = -25 },
  { frame = 124, dx = 100, dy = -30 },
  { frame = 139, dx = 120, dy = -30 },
  { frame = 152, dx = 145, dy = -40 },
]

[transit]
predicted_start = "21:48:00"    # local predicted ingress (UTC wall clock)
predicted_end   = "23:36:00"    # local predicted egress
baseline_fit = "linear"         # "median" (legacy) | "linear" (R-13)

[system]                        # catalogue values, NASA Exoplanet Archive / exoplanet.eu
r_star = 0.79                   # R_sun
r_star_err = 0.02
semi_major_axis = 0.0272        # au
period = 1.74978                # days
m_planet = 0.46                 # M_jup
m_planet_err = 0.02
transit_duration = 110          # minutes
ra = "23h13m58.76s"             # target ICRS coords — needed for BJD_TDB (§6)
dec = "+08d45m40.6s"
[system.site]                   # observatory — needed for BJD_TDB (§6)
name = "Lustbuehel Observatory"
lat = 47.0678                   # deg
lon = 15.4936                   # deg
height = 480.0                  # m

[output]
write_csv = true
figures = true                  # figures optional and non-fatal (R-19, §10.2)
save_light_frames = false       # was create_lights
save_reduced_frames = false     # was create_reduced
colormap = "gray_r"
figsize = [15, 10]

[logging]
level = "INFO"                  # default INFO (R-17)
file = "exotransit.log"
```

Notes:

- The five mutually-exclusive legacy flags (`no_red`, `dark_red`, `bias_red`,
  `dark_bias_red`, `bias_min`/`bias_sigma`) collapse into
  `reduction.method` + `reduction.cut` — illegal combinations are
  unrepresentable (R-8). Legacy mapping: `no_red` → `method="none"`;
  `dark_red` → `"standard"`; `bias_red` → `"bias"`; `dark_bias_red` →
  `"dark_bias"`; `bias_min` → `"bias"` + `cut="min"`; `bias_sigma` →
  `"bias"` + `cut="sigma_clip"`; `sigma_red` → `cut="sigma_clip"` (its dead
  `[0,1,1]` branch in the legacy code is retired).
- Example configs `configs/wasp52b.toml`, `configs/hatp19b.toml`,
  `configs/tres5b.toml` MUST be shipped, populated from the commented blocks
  in `exo_input_values.py` (R-7).
- Physical constants (R_Jup, R_sun, au, M_Jup) move into `planet.py` as
  `astropy.constants`/module constants — they are not configuration.
- The legacy `starttransit_pred`/`endtransit_pred` frame indices and
  `x_ax_loc` tick indices are retired: transit windows derive from
  `[transit]` times (§8.1) and plot ticks are computed from the time axis.
- The `option_compare` CSV workflow is retired as a config option; method
  comparison is a CLI concern (two runs + the comparison output of §9).

### S-5 Config loading & validation

**Traces:** R-6, R-18, ADR-0001

- Parse with stdlib `tomllib`; write (CLI `--emit-config`, web export) with
  `tomli-w`.
- Validation happens in `config.load(path) -> Config` before any data is
  touched. A malformed config MUST raise `ConfigError` naming the offending
  field, the received value, and the expectation, e.g.
  `[photometry].annulus_inner: got 8.0, must be < annulus_outer (6.0)`.
- Validation rules (minimum set):
  - `reduction.method` ∈ {none, standard, bias, dark_bias}; `cut` ∈ its enum.
  - `aperture_radius > 0`; `0 < annulus_inner < annulus_outer`.
  - `fwhm > 0`; `threshold_factor > 0`; `background_maxiters >= 1`.
  - `stars.calibrators` non-empty (N ≥ 1, R-15); star coordinates ≥ 0.
  - `crop_half_width >= ceil(annulus_outer)` (the annulus must fit the crop).
  - paths for the frame categories required by `reduction.method` exist and
    are non-empty directories (darks needed for standard/dark_bias, bias for
    bias/dark_bias) — checked at load, not at first use (R-18).
  - `ra`/`dec` parse as ICRS coordinates; site lat/lon/height in range.
  - unknown keys are rejected (typo protection), with the nearest valid key
    suggested.

## 3. Command-line interface

### S-6 CLI

**Traces:** R-5, R-9, R-17

```
exotransit reduce CONFIG.toml [--reduction METHOD] [--aperture R]
                              [--output-dir DIR] [--log-level LEVEL]
                              [--no-figures] [--dry-run]
exotransit validate CONFIG.toml          # parse + validate only, exit 0/2
```

- Overrides apply after config validation and are re-validated.
- `--dry-run` performs config validation, input discovery, and header checks
  (§4.3, §6), prints the run plan (frame counts, method, star list), and
  exits without computing.
- Exit codes: `0` success; `2` configuration/validation error; `3` data error
  (missing/inconsistent frames, header failures); `4` reduction error
  (e.g. all frames flagged, §5.4); `1` unexpected exception.
- `--log-level` sets the root `exotransit` logger (default `INFO`, R-17).

## 4. Calibration stage

### S-7 Master frame construction

**Traces:** R-10, R-18; fixes TODO "Honor combo_master_bias"

- `calibration.build_master(frames: np.ndarray, combine: Literal["mean","median"]) -> np.ndarray`
  stacks a frame cube and combines along the stack axis.
- Master dark uses `reduction.master_dark_combine`; master bias uses
  `reduction.master_bias_combine`. The legacy bug — `combo_master_bias`
  ignored, bias always mean-combined (`ExoplanetLightcurve.py:99`) — is fixed;
  the regression fixture pins the *documented* behaviour (median default).
- Masters are built once per run and passed via `MasterFrames` (S-2). Only the
  masters required by `reduction.method` are built.

### S-8 Frame calibration

**Traces:** R-8, R-14

Per light frame, in order:

1. Subtract masters per `reduction.method`:
   `none` → raw; `standard` → −dark; `bias` → −bias; `dark_bias` → −bias −dark.
2. **Flat-field** (R-14, P2, optional): if `paths.flats` is set, build a
   master flat (median combine of dark-subtracted flats, normalised to its
   median) and divide: `calibrated = subtracted / master_flat`. Off by
   default; absence of flats with the switch off is not an error.
3. Apply the per-frame `cut` (median/average/min pedestal subtraction or
   sigma-clip) on the *cropped* window, as today.

### S-9 Input validation

**Traces:** R-18

Before reduction starts, `io_fits.discover(cfg) -> FrameSet` MUST verify:

- each required directory exists and contains ≥ 1 FITS file — **no**
  auto-creation of missing input directories (the legacy `os.makedirs` on
  input paths is removed); only `paths.output` is created on demand;
- all frames within a category share dimensions; lights match dark/bias/flat
  dimensions;
- required header keywords are present on every light frame: `DATE-OBS` (or
  `TIME-OBS` per §6 fallback), `EXPTIME`;
- lights sort deterministically by observation time (not directory order).

Failures raise `DataError` listing every offending file, not just the first.

## 5. Photometry stage

### S-10 API modernisation

**Traces:** R-10

Port to current `astropy`/`photutils`, preserving numerics:

| Legacy (2018) | Current |
|---|---|
| `sigma_clip(..., iters=n)` | `sigma_clip(..., maxiters=n)` |
| `sigma_clipped_stats(..., iters=n)` | `sigma_clipped_stats(..., maxiters=n)` |
| `apertures.area()` / `annulus.area()` | `.area` property |
| `from photutils import DAOStarFinder, ...` | `from photutils.detection import DAOStarFinder`; `from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry`; `from photutils.detection import find_peaks` |
| `positions = (xcentroid, ycentroid)` column tuple | `np.transpose([xcentroid, ycentroid])` position array |
| masked-array leakage from `sigma_clip` into photometry | explicit `.filled(np.nan)` / documented mask policy |

The ported code MUST reproduce the acceptance invariant (§11.3) — this is the
regression gate for the port, verified before any behavioural change lands.

### S-11 Per-frame photometry algorithm

**Traces:** R-10, R-11, ADR-0004

For each star and frame (inside `measure_star`, S-2):

1. Crop a `(2·crop_half_width)²` window at the star's coordinates shifted by
   the frame's registration offset (§5.3).
2. Apply the per-frame cut (§4, step 3).
3. Background stats: `sigma_clipped_stats(window, sigma=background_sigma, maxiters=background_maxiters)`.
4. Detect: `DAOStarFinder(fwhm=fwhm, threshold=threshold_factor · stat)` where
   `stat` is the mean/median/std per `threshold_stat`.
5. **Detection guard (R-11):** select the detected source nearest the window
   centre. If zero sources → flag `NO_SOURCE`; if the nearest source is
   > `fwhm · 2` px from centre → flag `OFF_CENTER`; if ≥ 2 sources within
   `fwhm · 2` px of centre → flag `AMBIGUOUS`. Flagged frames get `flux = NaN`
   and are excluded downstream (§7, §8) — never a silent `sources[0]`.
6. Aperture photometry: `CircularAperture(r=aperture_radius)` +
   `CircularAnnulus(annulus_inner, annulus_outer)` at the selected centroid,
   `method=cfg.method, subpixels=cfg.subpixels`.
7. Local sky subtraction:
   `flux = aperture_sum − (annulus_sum / annulus.area) · aperture.area`.

### S-12 Drift tracking

**Traces:** R-16, ADR-0004

- `tracking.mode = "auto"` (default): register each light frame against
  `reference_frame` by **phase cross-correlation**
  (`skimage.registration.phase_cross_correlation`, `upsample_factor >= 10`
  for sub-pixel shifts) on the *full calibrated frame*, yielding one global
  `(dx, dy)` per frame applied to every star's crop window. Per-star centroid
  refinement then happens inside the window (S-11 step 5/6).
- `mode = "manual"`: legacy behaviour — step-wise shifts from
  `manual_shifts` applied cumulatively from the given frame index onward.
- `mode = "off"`: fixed windows.
- The computed shift series is recorded in the outputs (§9) for provenance.
- Runtime: auto-registration MUST keep total runtime within NFR-1 (§12.1);
  frames MAY be 2×2-binned for the correlation only.

### S-13 Per-frame quality flags

**Traces:** R-11, R-23

`QualityFlag` is a small int-flag enum recorded per star and frame:
`OK`, `NO_SOURCE`, `OFF_CENTER`, `AMBIGUOUS`, `SATURATED` (any pixel in the
aperture at/above a configurable saturation level, default 65535),
`REGISTRATION_FAILED`. Flags propagate to the CSV (§9.1). If more than 50 % of
science-star frames are flagged, the run fails with exit code 4 (§3).

## 6. Time base

### S-14 BJD_TDB computation

**Traces:** R-12, R-18, ADR-0003

- Per frame: read `DATE-OBS` (+ `TIME-OBS` when the date keyword lacks the
  time part) and `EXPTIME`; build
  `t_mid = Time(date-obs, scale="utc", location=site) + EXPTIME/2`.
- Convert: `bjd_tdb = (t_mid.tdb + t_mid.light_travel_time(target_coord, kind="barycentric", location=site)).jd`
  with `target_coord` from `[system].ra/dec` and `site` from `[system.site]`.
- Both the raw header string (provenance) and BJD_TDB are carried in
  `FrameMeta` and written to the CSV (§9.1). BJD_TDB is the light-curve time
  ordinate everywhere (plots, windows, fits).
- Missing/unparseable `DATE-OBS`/`EXPTIME` or missing target/site data is a
  validation failure (R-18) at discovery time (§4.3), naming the frame and
  keyword.

## 7. Differential light curve

### S-15 Weighted ensemble comparison

**Traces:** R-15, ADR-0005

- Input: science `StarPhotometry` + N ≥ 1 calibrator `StarPhotometry`
  (flagged frames excluded pairwise).
- **N = 1:** the differential curve is simply `F_sci / F_cal1`; no weighting,
  no rejection (guarded path).
- **N ≥ 2 (Broeg-style):**
  1. Initial weights `w_k = 1/σ_k²`, where `σ_k` is calibrator *k*'s scatter
     over the **out-of-transit baseline window only** (§8.1) of its normalised
     flux — never the in-transit window (avoids circularity).
  2. Artificial comparison `C_i = Σ w_k · F_{k,i} / Σ w_k` (fluxes normalised
     to their baseline medians before combining).
  3. Iterate: recompute each calibrator's residual scatter against the
     current ensemble, update weights, and **drop** any calibrator whose
     removal lowers the out-of-transit residual RMS of `F_sci / C`; stop when
     membership and weights are stable (tolerance 1e-6, max 20 iterations).
  4. At least one calibrator always survives (fall back to the single best if
     rejection would empty the set).
- Outputs: the ensemble curve `F_sci / C`, per-calibrator diagnostic ratios
  `F_sci / F_k` (the legacy sci/cal1, sci/cal2 plots), final weights, and the
  used/rejected membership list — all recorded in §9.

## 8. Transit depth & planet parameters

### S-16 Transit windows & baseline fit

**Traces:** R-13, ADR-0003

- The three windows derive from `[transit].predicted_start/predicted_end`
  converted to BJD_TDB on the observation date:
  - **pre-transit baseline** = frames before predicted ingress,
  - **in-transit** = frames between ingress + and egress − 10 % of the
    transit duration (core, avoiding limb effects at contact points),
  - **post-transit baseline** = frames after predicted egress.
  This replaces the fixed "first 20 / middle 20 / last 20" frame windows; a
  window with < 5 usable frames is a warning, an empty one is an error.
- `baseline_fit = "linear"` (default): fit `a + b·t` to the combined pre+post
  baseline of the differential curve (the airmass slope seen for HAT-P-19 b)
  and evaluate the baseline at mid-transit; depth
  `ΔF/F = 1 − median(in-transit) / baseline(t_mid)` on the trend-normalised
  curve.
- `baseline_fit = "median"` (legacy): depth = mean of pre/post medians minus
  in-transit median, as today (`ExoplanetLightcurve.py:433`), kept for the
  regression comparison.

### S-17 Derived parameters

**Traces:** R-13, R-20; CLAUDE.md §2

With depth `d = ΔF/F`, catalogue `R_*`, `a`, `P`, `M_p`, transit duration
`t_dur`:

- magnitude depth `Δm = −2.5·log₁₀(1 − d)`;
- planet radius `R_p = √(d) · R_*` (in metres via R_sun; reported in R_Jup);
- density `ρ_p = 3 M_p / (4π R_p³)`;
- inclination from
  `cos i = √( ((R_* + R_p)/a)² − sin²(π·t_dur/P) )` (same relation as
  `ExoplanetLightcurve.py:453`, rewritten dimensionally consistently in SI —
  the refactor MUST document the exact formula it implements and its
  derivation in the `planet.py` docstring, R-25);
- error propagation (Gaussian) from `r_star_err` and `m_planet_err`, as
  today.

Each quantity gets a unit-tested reference value (§11.1).

### S-18 Depth uncertainty from photometric scatter

**Traces:** R-27, §9.3

The depth uncertainty additionally includes the measured scatter:
`σ_d² = σ_base²/n_base + σ_in²/n_in` from the baseline and in-transit residual
scatter of the differential curve, propagated into `R_p`, `ρ`, `i` alongside
the catalogue errors. Reported separately (`statistical` vs `catalogue`) and
combined in quadrature in §9.3.

### S-19 T₀ and O−C

**Traces:** R-28, ADR-0003, §9.3

Estimate mid-transit time `T₀` (BJD_TDB) as the midpoint of observed ingress/
egress crossings of the half-depth level (or, minimally, the flux-weighted
minimum of the smoothed in-transit curve). If the config provides a reference
ephemeris (`t0_ref`, `period`), report `O − C = T₀ − (t0_ref + E·P)` with the
nearest integer epoch `E`. Both go into the results JSON (§9.3).

### S-20 Limb darkening

**Traces:** R-29, §9.3

The estimator `R_p = √(d)·R_*` ignores stellar limb darkening and therefore
biases `d` (box-shaped transit assumption). Specification: (a) the results
JSON and README state this caveat explicitly; (b) an **optional** P2 stage
`exotransit fit` MAY fit a proper transit model (e.g. `batman` or
`pytransit`, quadratic limb-darkening law) to the differential curve,
reporting model-based `R_p/R_*`, `i`, `T₀` next to — not instead of — the
simple estimator. The simple estimator remains the default and the acceptance
gate.

## 9. Outputs

### S-21 Per-frame CSV

**Traces:** R-23, ADR-0006

`<output>/<casename>_lightcurve.csv`, one row per light frame:

```
frame_index, file, time_raw, bjd_tdb, exptime,
shift_dx, shift_dy,
flux_sci, quality_sci,
flux_cal_<name>..., quality_cal_<name>...,   # one pair per calibrator
in_ensemble_<name>...,                       # true/false per calibrator
ratio_ensemble, ratio_<name>...              # sci/C and per-calibrator diagnostics
```

Quality columns hold the flag names of §5.4. The header row is stable and
documented; columns for calibrators appear in config order.

### S-22 JSON sidecar

**Traces:** R-23, R-24, R-27, R-28, R-30, ADR-0006

`<output>/<casename>_result.json`:

```jsonc
{
  "schema_version": 1,                    // R-30
  "target": "WASP-52 b",
  "provenance": {                          // R-24
    "software": "exotransit", "version": "…", "python": "…",
    "config_path": "…", "config_sha256": "…",
    "input_files": { "lights": […], "darks": […], "bias": […], "flats": [] },
    "run_started_utc": "…", "run_finished_utc": "…"
  },
  "reduction": { "method": "standard", "cut": "none",
                  "tracking_mode": "auto", "baseline_fit": "linear" },
  "ensemble": { "used": ["Calibrator_1", "Calibrator_2"], "rejected": [],
                 "weights": { "Calibrator_1": 0.6, "Calibrator_2": 0.4 } },
  "quality": { "n_frames": 0, "n_flagged_sci": 0, "flag_counts": {} },
  "results": {
    "depth": { "value": 0.0, "err_statistical": 0.0 },        // §S-18 (R-27)
    "delta_mag": 0.0,
    "r_p_rjup": { "value": 0.0, "err_catalogue": 0.0,
                   "err_statistical": 0.0, "err_total": 0.0 },
    "density_kg_m3": { "value": 0.0, "err_total": 0.0 },
    "inclination_deg": { "value": 0.0, "max": 0.0 },
    "t0_bjd_tdb": null, "o_minus_c_days": null                 // §S-19 (R-28)
  }
}
```

CSV + JSON together MUST be sufficient to regenerate every figure (R-23).

### S-23 Figures

**Traces:** R-19

All current figures are preserved (master frames, optional light/reduced
frames, per-star flux, sci/ensemble and per-calibrator ratio curves with
transit markers), generated by `plots.py` from `RunResult` only. Figure
generation is optional (`output.figures`) and wrapped so that any plotting
exception logs a warning and never aborts the run after fluxes are computed
(R-19). The x-axis uses BJD_TDB (with a UTC secondary labelling), removing the
legacy `x_ax_loc` fixed tick indices.

## 10. Logging & robustness

### S-24 Logging

**Traces:** R-17

- All `print` diagnostics become `logging` calls on the `exotransit.*`
  hierarchy: per-frame details at `DEBUG`, per-stage summaries at `INFO`,
  recoverable oddities (flagged frames, dropped calibrators, plot failures)
  at `WARNING`, aborts at `ERROR`.
- The broken legacy calls (`logger.info('Min:', value)` with stray positional
  args, `ExoplanetLightcurve.py:154`) are replaced by f-string messages.
- Handlers are configured only in `cli.main()` / `pipeline.run()`, never at
  import (R-1). Level from config, overridable by `--log-level`.

## 11. Testing & CI

### S-25 Unit tests

**Traces:** R-20, ADR-0008

`tests/` (pytest), targeting per module — with synthetic data where
determinism helps:

| Module | Cases (minimum) |
|---|---|
| `config` | valid load; every validation rule of S-5 rejects with the named field |
| `calibration` | mean/median master combine on known cubes; each reduction method's arithmetic; flat normalisation |
| `photometry` | aperture/annulus sky subtraction against a hand-computed synthetic star; detection guard paths (zero/off-centre/ambiguous) |
| `tracking` | recovery of a known injected `(dx, dy)` shift to < 0.1 px; manual-shift application |
| `timebase` | BJD_TDB of a known frame against an independently computed reference (< 1 s) |
| `lightcurve` | N = 1 passthrough; weighted ensemble on synthetic curves with one variable calibrator → it is rejected |
| `planet` | `R_p`, `ρ`, `i`, error propagation vs hand-computed values from the WASP-52 b numbers |
| `outputs` | CSV/JSON schema round-trip; config hash stability |

### S-26 Regression fixture & acceptance test

**Traces:** R-21, ADR-0008

- Commit `tests/data/wasp52b_subset/` — a **down-sampled/cropped real
  WASP-52 b subset** (lights + darks + bias, a few tens of frames, cropped
  around the three stars) small enough for the repo, licensing confirmed
  compatible with GPL v3.
- `test_acceptance_wasp52b`: run the full pipeline
  (`method="standard"`, legacy `baseline_fit="median"`, manual tracking with
  the thesis shifts) on the fixture and assert the derived parameters within
  tolerances **frozen from the first successful run of the ported code on the
  fixture** (the fixture subset need not reproduce the full-dataset numbers
  exactly; the full-dataset gate R_p ≈ 1.15 R_Jup, ρ ≈ 400 kg/m³, i ≈ 87.3°
  is verified once manually and documented in the test).
- Any PR that shifts the acceptance numbers beyond tolerance fails CI.

### S-27 CI & toolchain

**Traces:** R-22, NFR-2, ADR-0002, ADR-0007

- GitHub Actions on push + PR: `uv sync` → `ruff check` + `ruff format
  --check` → `mypy exotransit/` (non-strict) → `pytest`.
- Ruff configured in `pyproject.toml`; its version pinned via `uv.lock`.
- Matrix: the oldest supported (3.13) and latest stable Python.

## 12. Non-functional

### S-28 Performance, typing, platform

**Traces:** NFR-1, NFR-2, NFR-3

- Wall-clock for a single-target reduction on the same data MUST NOT exceed
  the legacy script by more than ~20 % (auto-tracking included); measured on
  the regression fixture in CI as a soft (logged, non-failing) benchmark.
- Public functions carry type hints; `mypy` non-strict passes (CI-gated).
- Python ≥ 3.13; no OS-specific paths (use `pathlib`).

---

# Part II — Web data-input application (milestone level)

Deep UI/endpoint specification is deferred to a later pass; this section fixes
the architecture, the milestone contracts, and the acceptance criteria so Part
II work can be scheduled without re-reading the requirements.

### S-29 Architecture

**Traces:** W-NFR-4, W-NFR-5, ADR-0009, ADR-0010, ADR-0011

- **Backend:** FastAPI service importing `exotransit` directly (one science
  implementation, W-NFR-5/W-NFR-3). Pydantic models mirror the TOML schema of
  S-4 field-for-field so CLI and web validate identically (S-5 rules).
- **Frontend:** React SPA; custom canvas viewer rendering backend-produced
  PNG tiles (scaling linear/log/zscale server-side; display down-sampled,
  photometry full-resolution).
- **Interaction model:** guided wizard (ADR-0011) with steps
  *open data → check → pick stars → set options → enter system facts →
  review → save/run → results*, each step gated on validity (W-16), freely
  revisitable (W-17), pre-filled with the S-4 defaults (W-19), resumable via
  persisted sessions (W-3/W-18).
- **Deployment:** single-user local-first (`uvicorn` + browser), no auth
  (W-NFR-4).

### S-30 Milestone M1 — Config generator

**Traces:** W-1, W-2, W-4, W-5, W-8, W-9, W-11, W-14, W-15, W-16, W-19, W-21, W-NFR-1

- Endpoints: `POST /session` (create; upload or server-path source, W-1),
  `GET /session/{id}/summary` (per-category frame count, dimensions,
  exposure, `TIME-OBS` range, mismatch flags — reusing S-9 validation, W-2),
  `GET /frames/{i}/tile?scale=`, `POST /stars` (click → pixel coords, W-5),
  `GET/PUT /config` (validated against S-5), `GET /config/export` → TOML
  download (W-11), re-loadable.
- Wizard covers open-data → check → pick-stars → options → review → export;
  the export button is disabled while any step is invalid (W-9/W-16); a
  review screen lists every value before export (W-21).
- Timeline scrubber under the viewer with frame index + timestamp, keyboard
  stepping, pre-cached/down-sampled frames (W-14, W-NFR-2).
- Acceptance: a prepared user goes from a fresh frame directory to a valid
  exported TOML in a few minutes, zero hand-edited text (W-NFR-1); the
  exported TOML runs unmodified through `exotransit reduce`.

### S-31 Milestone M2 — Guided photometry

**Traces:** W-6, W-7, W-10, W-17, W-20

- Click-to-centroid snap (`POST /centroid`) with aperture + annulus overlay
  at the configured radii (W-6).
- Drift assistance: step frames on the timeline, mark index + new position to
  build `manual_shifts` visually, or preview auto-tracking shifts from S-12
  (W-7).
- System-parameter form with catalogue lookup (NASA Exoplanet Archive by
  target name) auto-filling `[system]`, manual override always possible
  (W-10). Lookup failures degrade gently to manual entry (W-20).

### S-32 Milestone M3 — Run & results

**Traces:** W-12, W-13, W-22, W-NFR-3

- `POST /run` executes `pipeline.run()` as a background job; `GET /run/{id}`
  streams progress; results view shows the differential curve(s) and the S-22
  parameter block (W-12).
- Live preview (W-22): parameter changes after a first result trigger a
  down-sampled re-preview of the affected artefact (overlay and/or curve);
  the saved/final run always recomputes at full resolution (W-NFR-3).
- Optional side-by-side comparison of two reduction methods + CSV download
  (W-13, replacing the legacy `option_compare` workflow).

---

# 13. Promoted additions — now in REQUIREMENTS.md

Surfaced while writing this specification and, as of **2026-07-04**, **promoted**
into `REQUIREMENTS.md` as requirements R-27–R-30. Retained here as a record of
origin; the backing specs (S-18, S-19, S-20, S-4, S-22) are now fully traced,
and no `Status: proposed` items remain.

1. **§13.1 Depth uncertainty from photometric scatter** (spec S-18). Today's
   error propagation uses only catalogue errors (`e_rstar`, `e_m_planet`);
   the dominant error source — the measured scatter of the light curve —
   never enters the reported uncertainties. Cheap to add and makes the error
   bars honest. *Promoted: R-27 (P1).*
2. **§13.2 T₀ and O−C output** (spec S-19). ADR-0003 adopts BJD_TDB explicitly
   "to enable proper T₀ and O−C", but no requirement asked for either.
   *Promoted: R-28 (P2).*
3. **§13.3 Limb-darkening caveat + optional transit-model fit** (spec S-20).
   The √depth estimator systematically underestimates R_p because limb
   darkening deepens the observed minimum; a batman/pytransit fit is the
   standard remedy and would modernise the science without touching the
   default path. *Promoted: R-29 (P2).*
4. **§13.4 `schema_version` field** (specs S-4, S-22) in the TOML config and
   the results JSON, so future format changes are detectable and migratable —
   cheap now, painful to retrofit once web-app sessions and saved configs
   exist. *Promoted: R-30 (P1).*

---

# 14. Traceability matrix

| Requirement / decision | Spec section(s) |
|---|---|
| R-1 | S-1, S-24 |
| R-2 | S-1 |
| R-3 | S-1, S-2 |
| R-4 | S-3 |
| R-5 | S-3, S-6 |
| R-6 | S-4, S-5 |
| R-7 | S-4 |
| R-8 | S-4, S-8 |
| R-9 | S-6 |
| R-10 | S-7, S-10, S-11 |
| R-11 | S-11, S-13 |
| R-12 | S-14 |
| R-13 | S-16, S-17 |
| R-14 | S-4, S-8 |
| R-15 | S-4, S-5, S-15 |
| R-16 | S-4, S-12 |
| R-17 | S-6, S-24 |
| R-18 | S-5, S-9, S-14 |
| R-19 | S-4, S-23 |
| R-20 | S-17, S-25 |
| R-21 | S-26 |
| R-22 | S-27 |
| R-23 | S-13, S-21, S-22 |
| R-24 | S-22 |
| R-25 | S-17 (planet.py docstring); full docstring pass tracked in TODO.md |
| R-26 | §14 (this matrix); CLAUDE.md anchors updated when modules land |
| R-27 | S-18 |
| R-28 | S-19 |
| R-29 | S-20 |
| R-30 | S-4, S-22 |
| NFR-1 | S-12, S-28 |
| NFR-2 | S-27, S-28 |
| NFR-3 | S-3, S-27, S-28 |
| W-1, W-2 | S-30 |
| W-3 | S-29 |
| W-4, W-5 | S-30 |
| W-6, W-7 | S-31 |
| W-8, W-9 | S-30 |
| W-10 | S-31 |
| W-11 | S-30 |
| W-12, W-13 | S-32 |
| W-14 | S-30 |
| W-15, W-16 | S-29, S-30 |
| W-17 | S-29, S-31 |
| W-18, W-19 | S-29 |
| W-20 | S-31 |
| W-21 | S-30 |
| W-22 | S-32 |
| W-NFR-1 | S-30 |
| W-NFR-2 | S-29, S-30 |
| W-NFR-3 | S-29, S-32 |
| W-NFR-4 | S-29 |
| W-NFR-5 | S-29 |
| ADR-0001 | S-4, S-5 |
| ADR-0002 | S-1, S-3, S-27 |
| ADR-0003 | S-14, S-16 |
| ADR-0004 | S-11, S-12 |
| ADR-0005 | S-15 |
| ADR-0006 | S-21, S-22 |
| ADR-0007 | S-27 |
| ADR-0008 | S-25, S-26 |
| ADR-0009 | S-29 |
| ADR-0010 | S-29 |
| ADR-0011 | S-29 |
| R-27–R-30 (promoted §13.1–§13.4) | S-18, S-19, S-20, S-4, S-22 |

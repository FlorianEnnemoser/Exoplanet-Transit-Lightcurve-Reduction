# Requirements

Engineering requirements for (I) refactoring the existing reduction pipeline and
(II) building a web application for fast data input. This document is a companion to
`CLAUDE.md`, which describes the current scientific methodology and code layout.

**Conventions.** Requirement keywords **MUST** / **SHOULD** / **MAY** follow
RFC 2119. Each requirement carries a stable identifier (`R-*` for the refactor,
`W-*` for the web app) and a priority: **P0** (blocking / correctness), **P1**
(important), **P2** (nice-to-have). Requirements describe *what* and *why*; they do
not prescribe a specific implementation unless correctness demands it.

**Baseline (today).** The pipeline is two modules — `ExoplanetLightcurve.py`
(a procedural script that runs on import, one reusable function `fluxtarget`) and
`exo_input_values.py` (configuration as code). It hard-wires one science target and
two calibrators, uses a ~2018 `photutils`/`astropy` API, tracks star drift by hand
via `i`/`shift_x`/`shift_y` arrays, performs no flat-fielding, and ships no tests,
CLI, packaging, or dependency manifest. Preserve the **GPL v3** licence throughout.

**Acceptance invariant.** Any refactor **MUST** preserve the thesis' headline result:
the *standard* (dark-subtraction) reduction of WASP-52 b reproduces catalogue values
(R_p ≈ 1.15 R_Jup, ρ ≈ 400 kg/m³, i ≈ 87.3°). "Does the standard branch still
reproduce catalogue values within uncertainty?" is the regression gate.

---

## Part I — Code refactoring requirements

### 1. Architecture & packaging

- **R-1 (P0).** The pipeline **MUST** be importable without executing. Move all
  top-level execution behind a `main()` guarded by `if __name__ == "__main__":`.
  Importing any module **MUST NOT** read files, create directories, or write output.
- **R-2 (P0).** Refactor the monolith into a small package (e.g. `exotransit/`) with
  separated concerns: I/O (`io_fits`), calibration (`calibration`), photometry
  (`photometry`), differential light curve (`lightcurve`), parameter derivation
  (`planet`), plotting (`plots`), and orchestration (`pipeline`).
- **R-3 (P1).** Replace hidden module-level state (global `images_array`,
  `dark_master`, `bias_master`) with explicit function arguments / return values or a
  small dataclass carrying run state. `fluxtarget` **MUST** take its master frames and
  image list as parameters rather than reading globals.
- **R-4 (P1).** Add a dependency manifest (`pyproject.toml` or `requirements.txt`)
  pinning `numpy`, `matplotlib`, `astropy`, `photutils`, and the Python version.
- **R-5 (P2).** Provide a console entry point (e.g. `exotransit reduce ...`) via
  `pyproject.toml` `[project.scripts]`.

### 2. Configuration & CLI

- **R-6 (P0).** Externalise configuration into a **data** file (YAML or JSON), not
  executable Python. Provide a schema and validation; a malformed config **MUST**
  fail fast with a clear message naming the offending field.
- **R-7 (P0).** Target selection **MUST** be a first-class input (a config file per
  target, or a `--target` selector) — not commenting/uncommenting code blocks.
  Ship the WASP-52 b, HAT-P-19 b, and TrES-5 b configurations as example files.
- **R-8 (P1).** The mutually-exclusive reduction flags **MUST** collapse into a single
  enumerated field (e.g. `reduction: standard | bias | dark_bias | none`) plus an
  optional per-frame cut, so illegal combinations are unrepresentable.
- **R-9 (P1).** Provide a CLI accepting a config path and overrides
  (`--reduction`, `--aperture`, `--output-dir`, `--dry-run`).

### 3. Scientific correctness & API modernisation

- **R-10 (P0).** Port the legacy API to current `photutils`/`astropy`:
  `iters=` → `maxiters=`; `CircularAperture.area()` / `CircularAnnulus.area()` method
  calls → `.area` property; and the `DAOStarFinder(...)(image)` call pattern to the
  current interface. The ported code **MUST** reproduce the acceptance invariant.
- **R-11 (P0).** Guard the source-detection step: if `DAOStarFinder` returns zero (or
  multiple ambiguous) sources for a frame, the frame **MUST** be flagged rather than
  silently indexing `sources[0]`. Emit a per-frame quality flag in the output.
- **R-12 (P1).** Read exposure time and timestamps from FITS headers robustly, and
  compute a proper time base (e.g. BJD or at least a continuous MJD) instead of the
  raw `TIME-OBS` string used only as a plot label.
- **R-13 (P1).** Replace the fixed "first 20 / middle 20 / last 20" transit-depth
  windows with windows derived from the predicted ingress/egress times, and implement
  the **linear baseline fit** for the out-of-transit trend (the airmass slope seen in
  HAT-P-19 b) that the thesis lists as future work.
- **R-14 (P2).** Add optional **flat-field** calibration (`Light − Dark) / Flat`,
  currently absent, behind a config switch.

### 4. Pipeline generalisation

- **R-15 (P1).** Generalise the star model from "1 science + exactly 2 calibrators"
  to *one science target and N ≥ 1 calibrators*, forming an ensemble comparison.
- **R-16 (P1).** Implement automatic **centroid tracking** (e.g. re-centroid the crop
  window on the brightest detected source each frame, or cross-correlate frames) to
  replace the manual `i`/`shift_x`/`shift_y` drift arrays — the acknowledged principal
  limitation. Keep manual shifts available as a fallback.

### 5. Robustness, logging, error handling

- **R-17 (P1).** Replace `print` diagnostics with structured `logging` at appropriate
  levels; make the log level a config/CLI option (default `INFO`).
- **R-18 (P1).** Validate inputs up front: directories exist and are non-empty, frames
  within a category share dimensions, required header keywords are present. Fail with
  actionable errors.
- **R-19 (P2).** Make figure generation optional and non-fatal (a plotting error
  **MUST NOT** abort the reduction after fluxes are computed).

### 6. Testing & CI

- **R-20 (P0).** Add a test suite (`pytest`) with unit tests for calibration,
  aperture/annulus sky subtraction, differential-ratio formation, and each derived
  quantity (R_p, ρ, i) against hand-computed values.
- **R-21 (P0).** Add a **regression test** on a small committed synthetic or
  down-sampled dataset asserting the WASP-52 b standard-reduction parameters within a
  tolerance (the acceptance invariant).
- **R-22 (P1).** Add CI (GitHub Actions) running the tests and a linter/formatter
  (`ruff`/`black`) on push and pull request.

### 7. Reproducibility & outputs

- **R-23 (P1).** Write a machine-readable results file (extend the current CSV, or add
  JSON) capturing per-frame flux, quality flags, the chosen reduction method, config
  hash, and derived parameters with uncertainties — enough to reproduce a figure.
- **R-24 (P2).** Record provenance in outputs (software version, config, input file
  list) so a run is self-describing.

### 8. Documentation

- **R-25 (P1).** Add docstrings (NumPy style) to all public functions and update
  `README.md` with install + run instructions once packaging lands.
- **R-26 (P2).** Keep `CLAUDE.md` `file:line` anchors current, or replace them with
  symbol references once the code is modularised.

### Non-functional (refactor)

- **NFR-1.** No change **SHOULD** increase single-target wall-clock runtime by more
  than ~20% versus the current script on the same data.
- **NFR-2.** Public APIs **SHOULD** carry type hints and pass `mypy` in non-strict mode.
- **NFR-3.** The package **MUST** run on a currently-supported Python (≥ 3.10).

---

## Part II — Web data-input application requirements

**Goal.** A browser application that makes the slowest, most error-prone parts of a
reduction — pointing at data, picking star pixel coordinates (today done manually in
SAOImage DS9), and tuning reduction/photometry parameters — **fast, visual, and
validated**, then emits a pipeline config (Part I, R-6) and/or launches a run.

**Primary user.** An observer/student reducing one transit at a time who wants to go
from a directory of FITS frames to a light curve with minimal typing.

### Functional requirements

**Session & data ingestion**

- **W-1 (P0).** The user **MUST** be able to start a reduction session by selecting a
  data source: uploading FITS frames, or pointing at server-side directories for
  lights, darks, and biases.
- **W-2 (P0).** The app **MUST** parse FITS headers on ingest and display a summary
  per category (frame count, dimensions, exposure time, `TIME-OBS` range) and flag
  mismatched dimensions or missing keywords before the user proceeds.
- **W-3 (P1).** The app **SHOULD** persist sessions so a user can reload a project and
  resume; sessions **SHOULD** be independent and isolated.

**Interactive frame viewer & fast coordinate entry (the core value)**

- **W-4 (P0).** The app **MUST** render a science frame in an interactive viewer with
  adjustable scaling (linear/log/zscale), zoom, and pan.
- **W-5 (P0).** The user **MUST** be able to designate the science target and each
  calibrator by **clicking** the star on the image; the click yields pixel coordinates
  that replace the DS9-picked `sci/cal*_coordinates` values. Selected stars **MUST** be
  labelled and listed, and be individually removable/renamable.
- **W-6 (P1).** On click, the app **SHOULD** snap to the local centroid and overlay the
  configured aperture and annulus, giving immediate visual feedback on radii choices.
- **W-7 (P1).** The viewer **SHOULD** let the user step through frames (slider/next)
  and, where drift is uncorrected, mark the frame index and new position to build the
  manual `i`/`shift_x`/`shift_y` arrays visually — or preview automatic tracking (R-16).

**Parameter forms & validation**

- **W-8 (P0).** The app **MUST** expose the reduction method as a single choice
  (none / standard / bias / dark+bias / bias+median), consistent with R-8, and the
  photometry parameters (`aperture`, `annulus` inner/outer, `fwhm`, detection
  threshold, `background_sigma`).
- **W-9 (P0).** All numeric inputs **MUST** be validated live (ranges, inner < outer
  annulus radius, aperture > 0) with inline errors; the "generate/run" action **MUST**
  be disabled while the form is invalid.
- **W-10 (P1).** The app **MUST** collect the system parameters needed for derivation
  (`rstar` ± error, `a`, `P`, `m_planet` ± error, transit duration, predicted
  ingress/egress). It **SHOULD** offer a catalogue lookup (NASA Exoplanet Archive /
  Exoplanet Transit Database) by target name to auto-fill these, with manual override.

**Output, run, and export**

- **W-11 (P0).** The app **MUST** export a valid pipeline configuration file (Part I
  schema) capturing every entered value, downloadable and re-loadable.
- **W-12 (P1).** The app **SHOULD** trigger a reduction run (locally or via a backend
  job) and display progress, then show the resulting differential light curve
  (`sci/cal1`, `sci/cal2`) and the derived planet parameters with uncertainties.
- **W-13 (P2).** The app **MAY** let the user compare two reduction methods
  side-by-side (mirroring the thesis' comparison and the existing `option_compare`
  CSV workflow) and download the results CSV.

### Non-functional (web app)

- **W-NFR-1 (P0, usability).** Reaching a runnable configuration from a fresh
  directory of frames **SHOULD** require no hand-editing of code or config text —
  clicks and validated form fields only. Target: a prepared user completes target +
  calibrator selection and parameters in a few minutes.
- **W-NFR-2 (P1, performance).** Frame rendering and click-to-centroid feedback
  **SHOULD** feel interactive (sub-second for a binned 1124×850 frame); large frames
  **SHOULD** be down-sampled for display while photometry uses full resolution.
- **W-NFR-3 (P1, correctness).** The config the app emits **MUST** drive the refactored
  pipeline to the *same* numerical result as entering those values by hand — the web
  layer is an input surface, not a second implementation of the science.
- **W-NFR-4 (P2, portability).** The app **SHOULD** run self-contained (local
  backend + browser) so it works on an observer's laptop without cloud services;
  authentication is out of scope for single-user local use.
- **W-NFR-5.** FITS parsing/photometry on the backend **SHOULD** reuse the Part I
  package directly rather than reimplementing it.

### Suggested (non-binding) shape

A thin backend exposing the refactored `exotransit` package (FITS ingest, frame
rendering as PNG/array tiles, centroiding, and run execution) behind a small HTTP API,
with a single-page frontend for the viewer and forms. Any stack meeting the functional
and non-functional requirements is acceptable; the requirements above are the contract.

### Milestones

1. **M1 — Config generator.** W-1, W-2, W-4, W-5, W-8, W-9, W-11: ingest frames, pick
   stars by click, fill validated forms, export a pipeline config. (No run yet.)
2. **M2 — Guided photometry.** W-6, W-7, W-10: centroid snapping, aperture overlay,
   drift/tracking assistance, catalogue auto-fill.
3. **M3 — Run & results.** W-12, W-13: execute the reduction and display/compare light
   curves and derived parameters.

# Architecture Decision Records

This file logs significant architecture and design decisions for the pipeline and its
planned refactor (see `REQUIREMENTS.md`). It uses a lightweight
[MADR](https://adr.github.io/madr/)-style format.

**Conventions.**
- Records are **append-only** and numbered in ascending order (`ADR-0001`, `ADR-0002`, …).
- **Status** is one of *Proposed*, *Accepted*, *Superseded* (name the superseding ADR),
  or *Deprecated*. A decision, once *Accepted*, is not edited in place — supersede it
  with a new record instead.
- Each record states **Context** (forces at play), **Decision**, and **Consequences**
  (positive `+` and negative `−`).

---

## ADR-0001 — Configuration file format: TOML

- **Status:** Accepted
- **Relates to:** `REQUIREMENTS.md` R-6, R-8, W-11

### Context

The refactor moves configuration out of executable Python (`exo_input_values.py`,
"configuration as code") into a validated **data** file so that behaviour is not
selected by editing/commenting source (see R-6, R-8). A format must be chosen. The
relevant forces:

- The config is **hand-edited by observers**, so comments and readability matter.
- Values are mostly **typed scalars and short arrays** (paths, radii, `[x, y]`
  coordinates, system parameters), not deeply nested structures.
- The target runtime is a **current Python (≥ 3.13)**, whose standard library ships
  `tomllib` for reading TOML — no dependency needed to parse.
- The format should have **low ambiguity** (YAML's implicit typing and significant
  whitespace are a known source of surprises; JSON forbids comments).

### Decision

Use **TOML** as the configuration file format for the refactored pipeline: one config
file per target, and the format the web data-input application (Part II, W-11) emits
and re-loads.

### Consequences

- `+` Read with the standard-library `tomllib` on Python ≥ 3.13 — zero parse dependency,
  no `tomli` backport needed.
- `+` Comments are supported (unlike JSON), aiding self-documenting target configs.
- `+` Explicit, unambiguous typing and no significant-whitespace pitfalls (vs YAML).
- `+` Maps cleanly onto the flat/lightly-nested parameter groups this pipeline needs.
- `−` `tomllib` is **read-only**; writing config (CLI/web export) needs a small writer
  dependency (e.g. `tomli-w`).
- `−` Deeply nested or richly structured data is less ergonomic in TOML than in YAML;
  acceptable here because the config is shallow.
- Supersedes the earlier "YAML or JSON" wording in `REQUIREMENTS.md` R-6.

---

## ADR-0002 — Packaging & dependency management: `uv` + `pyproject.toml`

- **Status:** Accepted
- **Relates to:** `REQUIREMENTS.md` R-4, R-5, R-22, NFR-2, NFR-3

### Context

R-4 requires a dependency manifest pinning `numpy`, `matplotlib`, `astropy`,
`photutils`, and the Python version, but deliberately leaves the format open
(`pyproject.toml` **or** `requirements.txt`). A concrete choice is needed. The
relevant forces:

- The refactor packages the code as `exotransit/` (R-2) with a console entry
  point (R-5). Both need PEP 621 `[project]` metadata (and `[project.scripts]`)
  that a bare `requirements.txt` cannot express — so `pyproject.toml` is required
  regardless.
- The scientific stack is **version-sensitive**: the legacy `astropy`/`photutils`
  API breakage that R-10 exists to fix is precisely this class of problem, so
  fully-pinned, reproducible environments across contributor laptops and CI
  (R-22) matter.
- Reproducibility wants a **lockfile**; fast, deterministic installs keep CI and
  onboarding cheap.
- The runtime target is a currently-supported Python (≥ 3.13, NFR-3), which
  should be easy to provision.

### Decision

Use **`uv`** as the project and dependency manager, with a single
**`pyproject.toml`** manifest as the source of truth: PEP 621 `[project]`
metadata, constrained dependencies, `requires-python = ">=3.13"` (NFR-3), a
committed **`uv.lock`** for reproducibility, development tools (`pytest`, `ruff`,
`mypy`) declared as a dependency group, and the console entry point via
`[project.scripts]` (R-5). No separate `requirements.txt`.

### Consequences

- `+` A single manifest satisfies R-4 and R-5; standard PEP 621 metadata yields a
  buildable, publishable package (R-2).
- `+` `uv.lock` gives reproducible, fully-pinned installs for CI (R-22) and
  contributors; `uv`'s resolver/installer is fast and can also provision the
  Python version itself (NFR-3).
- `+` Dependencies remain standard, so `pip install .` stays a viable fallback if
  `uv` is unavailable, and `mypy`/`ruff` config lives in the same file (NFR-2).
- `−` Contributors must install one new tool (mitigate: documented
  `pipx install uv` bootstrap; `uv` ships as a single static binary).
- `−` `uv` is comparatively young and `uv.lock` is uv-specific (mitigate: the
  `pyproject.toml` dependency declarations remain portable and tool-agnostic).
- Refines — does **not** supersede — R-4's "`pyproject.toml` or `requirements.txt`"
  wording to mandate `uv` + `pyproject.toml`.

---

## ADR-0003 — Photometric time base: BJD_TDB

- **Status:** Accepted
- **Relates to:** `REQUIREMENTS.md` R-12, R-18

### Context

The current code reads the `TIME-OBS` header as a **string used only as a plot
label** (`ExoplanetLightcurve.py:350`); there is no numeric, uniform time base.
Transit science — mid-transit time `T₀`, `O − C` against ephemerides, comparison
with catalogue values and other observers — needs a barycentric, relativistic-
clock-consistent scale. Forces: the data are single-site, but results should be
comparable to catalogues; `astropy` supplies `Time` + `SkyCoord.light_travel_time`
for BJD_TDB with no extra dependency.

### Decision

Compute **BJD_TDB per frame** from the FITS `DATE-OBS`/`TIME-OBS` and `EXPTIME`
(using **mid-exposure** time) plus the target coordinates, via
`astropy.time.Time` + barycentric `light_travel_time` in the **TDB** scale. Store
BJD_TDB as the light-curve time ordinate; retain the raw header timestamp for
provenance.

### Consequences

- `+` Community standard for transit timing; comparable across observatories and
  epochs; enables proper `T₀` and `O − C`.
- `+` `astropy`-native — no new dependency; also removes the midnight-rollover
  ambiguity of the raw wall-clock string.
- `+` Mid-exposure timing accounts for `EXPTIME`, improving accuracy.
- `−` Depends on reliable header keywords (`DATE-OBS`, `EXPTIME`) and target
  coordinates; must validate up front with a clear failure (ties to R-18) — a
  wrong site/coords yields subtly wrong times.
- `−` Marginally more per-frame computation (negligible).

---

## ADR-0004 — Automatic drift tracking: whole-frame cross-correlation registration

- **Status:** Accepted
- **Relates to:** `REQUIREMENTS.md` R-16, R-11

### Context

The acknowledged **principal limitation** is manual drift correction via
`i`/`shift_x`/`shift_y` arrays re-tuned per dataset
(`ExoplanetLightcurve.py:244`). R-16 requires automatic tracking. The candidates
were per-frame re-centroiding on the brightest source versus whole-frame
registration. Notably, KELT-16 b **failed** when out-of-focus early frames made
`DAOStarFinder` lock onto background — a single-source tracker is fragile in
exactly that regime, whereas a field-wide registration uses all the flux.

### Decision

Register each frame to a reference frame by **phase cross-correlation**
(e.g. `skimage.registration.phase_cross_correlation`, or an FFT-based equivalent),
producing a global `(dx, dy)` shift applied to every crop window; then refine
per-star centroids within the shifted window for photometry. Keep the manual
shift arrays available as an explicit **fallback** behind a config switch.

### Consequences

- `+` Uses whole-field signal → robust when any single star is poorly detected
  (the KELT-16 failure mode); one shift serves all stars consistently.
- `+` Eliminates the principal manual step; reproducible across datasets.
- `+` Manual fallback retained for pathological data.
- `−` Assumes translational drift (no significant rotation/scale); rotation would
  need a richer model — acceptable for this instrument/mount.
- `−` Needs a reference frame and sub-pixel interpolation, and adds a dependency
  (`scikit-image` or equivalent) — pin it via ADR-0002.
- `−` Per-frame FFT cost; must stay within the NFR-1 (~20% runtime) budget.

---

## ADR-0005 — Differential comparison: variance/SNR-weighted ensemble with iterative rejection

- **Status:** Accepted
- **Relates to:** `REQUIREMENTS.md` R-15, R-11, R-13

### Context

Generalising from exactly two calibrators to *N ≥ 1* (R-15) requires a
combination rule. An equal-weight sum maximises Poisson SNR but does **not**
suppress variable or systematic-laden comparison stars. Both current best-practice
tooling and post-2021 pipelines favour a **weighted artificial comparison** with
iterative rejection of unsuitable stars; this is a deliberate, literature-grounded
choice (the user requested current references).

### Decision

Construct an **artificial comparison star** as the weighted sum of calibrator
fluxes, with weights derived self-consistently from each star's **out-of-transit
scatter / SNR**, iterating to drop stars that *increase* the residual RMS/BIC
(a Broeg-style algorithm). Form `science / artificial-comparison` as the
differential light curve, retain individual `sci/cal` ratios as diagnostics, and
record which comparisons were used/rejected in the outputs (ties R-23). Weighting
uses **only** the baseline (out-of-transit) window to avoid circularity with the
transit-depth windows (R-13); guard the `N = 1` case (no rejection).

### Method references

- Broeg, Fernández & Neuhäuser (2005), *Astron. Nachr.* **326**, 134 — origin of
  the optimum artificial-comparison algorithm (self-consistent variability
  weights). DOI [10.1002/asna.200410350](https://doi.org/10.1002/asna.200410350)
- Collins, Kielkopf, Stassun & Hessman (2017), *AJ* **153**, 77 — AstroImageJ,
  the weighted-ensemble implementation that is the AAVSO/ExoClock reference tool.
  DOI [10.3847/1538-3881/153/2/77](https://doi.org/10.3847/1538-3881/153/2/77)
- Dransfield et al. (2022), *Proc. SPIE* **12186**, 121861F — ASTEP+ automatic
  weighted differential-photometry pipeline (post-2021).
  DOI [10.1117/12.2629920](https://doi.org/10.1117/12.2629920)

### Consequences

- `+` Higher precision than an equal-weight sum; suppresses variable comparisons;
  matches community-standard tooling.
- `+` Scales to N calibrators and degrades gracefully to a single comparison.
- `+` The rejection log improves transparency and reproducibility.
- `−` More complex: needs per-star noise estimates and an iteration/convergence
  criterion; the `N = 1` path must be handled explicitly.
- `−` Weights depend on the out-of-transit window definition (coupled to R-13);
  mitigated by weighting on baseline only.
- `−` Must still satisfy the **acceptance invariant**: the WASP-52 b weighted
  result must reproduce catalogue values within uncertainty.

---

## ADR-0006 — Results output: extended CSV + JSON sidecar

- **Status:** Accepted
- **Relates to:** `REQUIREMENTS.md` R-23, R-24, R-11

### Context

Output today is a flat CSV of `NUMBER, TIME-OBS, FLUX-*`
(`ExoplanetLightcurve.py:356`). R-23 wants a machine-readable results file
capturing per-frame flux, quality flags, chosen method, config hash, and derived
parameters with uncertainties, plus provenance (R-24). Per-frame tabular data
suits CSV; nested run metadata and uncertainties suit JSON.

### Decision

Write **both** per run: a per-frame **CSV** (extended with BJD_TDB, per-frame
quality flags, and comparison-ensemble membership) for quick inspection, and a
**JSON sidecar** holding run metadata — config hash, software/version provenance
(R-24), input file list, reduction method, and derived `R_p`, `ρ`, `i` with
uncertainties.

### Consequences

- `+` Human-friendly rows (CSV) alongside machine-friendly nested metadata (JSON);
  together they are enough to reproduce a figure.
- `+` Config hash + provenance make each run self-describing (R-24).
- `+` JSON needs only the standard-library `json` writer — no new dependency.
- `−` Two files to keep schema-consistent; the schema must be documented.

---

## ADR-0007 — Lint/format toolchain: Ruff

- **Status:** Accepted
- **Relates to:** `REQUIREMENTS.md` R-22, NFR-2

### Context

R-22 requires CI linting and formatting. Options: Ruff (lint + format),
Black + Ruff, or Black + flake8. The project already uses `uv` + `pyproject.toml`
(ADR-0002), which favours a single fast tool configured in one file.

### Decision

Use **Ruff** for both linting and formatting, configured in `pyproject.toml` and
enforced in CI (R-22). Keep **mypy** separate for type checking (NFR-2).

### Consequences

- `+` One fast tool, a single config section, fewer dependencies; aligns with
  ADR-0002.
- `+` Ruff's formatter is Black-compatible, so style is familiar and migration is
  trivial.
- `−` Ruff moves quickly — pin its version in `uv.lock` to keep CI stable.
- `−` A few niche flake8 plugins have no Ruff equivalent; acceptable here.

---

## ADR-0008 — Regression fixture: down-sampled real WASP-52 b subset

- **Status:** Accepted
- **Relates to:** `REQUIREMENTS.md` R-21, R-20 (acceptance invariant)

### Context

The acceptance invariant requires the WASP-52 b *standard* reduction to reproduce
catalogue values (`R_p ≈ 1.15 R_Jup`, `ρ ≈ 400 kg/m³`, `i ≈ 87.3°`). R-21 needs a
small committed dataset to guard it. Options: synthetic frames, a down-sampled
real subset, or both.

### Decision

Commit a small **down-sampled/cropped real WASP-52 b** subset (lights plus
darks/bias) sufficient to run the standard reduction end to end and assert the
derived parameters within tolerance. Reserve **synthetic** frames for isolated
unit tests (R-20) where determinism helps — not for the acceptance gate.

### Consequences

- `+` Most faithful gate; exercises real FITS headers, timestamps, and detection
  on real PSFs, and directly encodes the thesis' headline result as the contract.
- `−` Real frames grow the repo (mitigate: crop/bin and keep only a few frames);
  confirm data licensing permits committing them under GPL v3.
- `−` A real subset may miss specific transit-window edge cases → complement with
  targeted synthetic unit tests where useful.

---

## ADR-0009 — Web backend framework: FastAPI wrapping `exotransit`

- **Status:** Accepted
- **Relates to:** `REQUIREMENTS.md` W (Part II), W-NFR-5, W-NFR-3, W-11, W-12, W-3

### Context

The web data-input app (Part II) must **reuse** the Part I `exotransit` package
rather than reimplement the science (W-NFR-5, W-NFR-3), and expose FITS ingest,
tile rendering, centroiding, config export (W-11), and run execution (W-12) for a
single-user, local-first deployment (W-NFR-4). Options: FastAPI, Flask, Django.

### Decision

Use **FastAPI** as a thin backend exposing `exotransit` behind a small HTTP API:
FITS ingest/summary, frame-tile rendering, click-to-centroid, config
generation/validation, and run execution. Pydantic models mirror the TOML config
schema (ADR-0001) so validation is shared between CLI and web.

### Consequences

- `+` Typed, async, minimal; reuses Part I directly (W-NFR-5) → one implementation
  of the science (W-NFR-3).
- `+` Pydantic validation aligns with the config schema (R-6/ADR-0001); OpenAPI
  docs come for free.
- `+` Lightweight and local-first (W-NFR-4).
- `−` Async adds complexity for long-running reductions (mitigate: a background
  task/job endpoint for W-12).
- `−` Not batteries-included (no ORM/admin) — fine for a single-user tool; session
  persistence (W-3) uses lightweight storage.

---

## ADR-0010 — Web frontend & FITS viewer: React SPA with server-rendered canvas tiles

- **Status:** Accepted
- **Relates to:** `REQUIREMENTS.md` W-4, W-5, W-6, W-14, W-NFR-2, W-NFR-3

### Context

The core value is an interactive frame viewer: scaling/zoom/pan (W-4),
click-to-pick stars yielding pixel coordinates (W-5), centroid snap plus
aperture/annulus overlay (W-6), and a timeline scrubber synchronised with the
light curve (W-14) — all staying interactive (W-NFR-2). Options: a custom
React + canvas viewer fed by backend tiles, an embedded JS9, or Aladin Lite.

### Decision

Build a **React single-page app** with a custom **canvas/WebGL viewer** that
renders PNG/array tiles produced by the FastAPI backend (ADR-0009). The backend
applies scaling (linear/log/zscale) and **down-samples for display** (W-NFR-2)
while photometry uses **full resolution** (W-NFR-3); the frontend owns
click-to-pick, overlays, and the timeline synced to the light-curve view.

### Consequences

- `+` Full control of the bespoke UX (pixel picking, overlays, timeline ↔ light
  curve sync) that off-the-shelf viewers make awkward.
- `+` Backend-side rendering/down-sampling keeps scrubbing interactive on large
  frames (W-NFR-2) while the science stays full-resolution (W-NFR-3).
- `+` Reuses the backend's `astropy`/`exotransit` stack for scaling/tiling — no
  second FITS implementation in JavaScript.
- `−` More frontend code than embedding a ready viewer (JS9/Aladin); we build
  zoom/pan/scaling UI ourselves.
- `−` Tile round-trips need caching/pre-fetch to feel instant (mitigate: pre-cache
  adjacent frames, as W-14 suggests).
---

## ADR-0011 — Web app interaction model: guided wizard with live preview

- **Status:** Accepted
- **Relates to:** `REQUIREMENTS.md` W-15–W-22, W-6, W-9, W-NFR-1, W-NFR-2

### Context

The web app's primary user is a beginner reducing one transit at a time who wants
minimal typing (W-NFR-1). The interface can be shaped in three broad ways:

- **Guided wizard** — leads the user through the steps one at a time (open data →
  check → pick stars → set options → enter facts → review → save/run → results).
- **Free-form page** — everything on a single screen, done in any order. Fast for an
  expert; easy for a beginner to get lost or skip a step.
- **Node-graph editor** — a visual canvas of boxes ("nodes"), one per processing
  step, wired together; the user clicks a node to tune its settings and watches the
  result update live (like the node editors in photo/video tools). Powerful and a
  natural fit for live tuning, but the most to build and the most concepts for a
  newcomer to learn.

A strongly desired quality — raised by the maintainer — is **live feedback**:
change a parameter and immediately see the effect on the result.

### Decision

Use a **guided step-by-step wizard** as the default and primary interface,
implementing the User-workflow requirements (W-15–W-22). Build **live preview**
into it (W-22): once a first result exists, adjusting a photometry/reduction
parameter updates a quick preview (frame overlay and/or light curve). Record the
**node-graph editor** as a considered option, **deferred** as an optional future
"advanced mode" rather than the first interface.

### Consequences

- `+` Beginner-friendly, linear path; hard to get lost or skip a required step
  (W-15/W-16); directly satisfies the "few minutes, clicks only" target (W-NFR-1).
- `+` Live preview delivers the "change a knob, see the effect" benefit the
  maintainer wants **without** the cost of a full node-graph editor.
- `+` Wizard steps map cleanly onto the milestones (M1–M3) and onto the backend
  endpoints (ADR-0009), keeping frontend and backend aligned.
- `−` A fixed wizard can feel restrictive to an expert doing many reductions
  (mitigate: allow jumping between already-completed steps, W-17; an advanced/
  free-form mode can be added later).
- `−` Live preview needs fast, possibly down-sampled recomputation (W-NFR-2) and
  careful state handling; previews stay approximate while the final saved/run
  result uses full resolution (W-NFR-3).
- `−` The node-graph editor is deferred, not adopted: it is powerful and excellent
  for live tuning, but significantly larger to build and steeper for a beginner.
  Revisit only if a power-user "advanced mode" is warranted; the wizard + live
  preview captures most of its value first.

---

## ADR-0012 — Web preview: in-process background job reusing the pipeline stages

- **Status:** Accepted
- **Relates to:** `REQUIREMENTS.md` W-12, W-22; ADR-0009, ADR-0011

### Context

The Preview step (wizard step 7) runs the reduction far enough to show the tracked
target and its differential light curve **before** the user commits to a full run.
Photometering a whole night (~150–300 frames) takes seconds to a minute, so it
cannot block the request thread, and the wizard needs progress feedback. Options:

- **Synchronous request** — one `POST` that returns when done. Simplest, but a
  multi-second hang with no progress and a fragile long-lived HTTP connection.
- **In-process background thread + poll** — start a job, poll status/progress.
- **External task queue** (Celery/RQ + broker) — robust, but a heavy dependency and
  deployment burden for a local single-user tool (W-NFR-4/5).

A second question is *what* the preview computes: re-implementing the reduction
would fork the science (rejected on sight, ADR-0009's single-source-of-truth rule).

### Decision

Run the preview as an **in-process background thread** (`webapp/server/preview.py`)
with a module-level job store keyed by session id, exposed as
`POST /api/sessions/{sid}/preview` (start, idempotent while running) and
`GET …/preview` (status/progress/result). The worker **reuses the
`exotransit.pipeline.run` stage functions** (`config.load → io_fits.discover →
calibration.build_masters → tracking → photometry.measure_star → lightcurve.differential`)
up to the light curve only — no `planet.compute`, no file outputs — and reports
progress per measured star.

### Consequences

- `+` No new dependency (stdlib `threading`); reuses every science stage, so the
  preview and a full run agree by construction (W-22, ADR-0009).
- `+` Non-blocking with real progress; the invariant-pinned `pipeline.run` is left
  untouched (the ~10-line median/linear branch is deliberately duplicated).
- `−` The job store is per-process and lost on restart, and there is no cross-process
  scaling — acceptable for the local single-user tool; revisit with a real queue only
  if multi-user/hosted deployment is ever wanted.
- `−` Progress granularity is per-star, not per-frame (a per-frame callback into
  `measure_star` is the noted upgrade path).

---

## ADR-0013 — NASA Exoplanet Archive lookup over stdlib TAP (no `astroquery`)

- **Status:** Accepted
- **Relates to:** `REQUIREMENTS.md` W-10, W-20

### Context

The System step offers a "look up by target name" button that fills stellar/planet
parameters from the NASA Exoplanet Archive (W-10), degrading gently on failure
(W-20). The canonical client library is **`astroquery`**, but it is a large
dependency (pulls in much of the astropy affiliated stack) for a single query, and
the web feature only needs one composite-parameter row.

### Decision

Query the Archive's **TAP sync service** directly over stdlib `urllib`
(`webapp/server/archive.py`), selecting the needed columns from the `pscomppars`
composite table by exact `pl_name`, and map them onto the `[system]` fields (with
`pl_trandur` hours → minutes). Any network/parse error or empty result returns an
empty mapping plus a human note — never an exception (W-20).

### Consequences

- `+` No new dependency; one small, self-contained module and endpoint.
- `+` Failures are non-fatal and manual entry always remains available (W-20).
- `−` Exact-name match only (no fuzzy resolver/name aliasing that `astroquery` would
  provide) — the noted upgrade path if name-matching friction shows up.
- `−` Couples to the Archive's TAP schema/column names; a schema change needs a code
  update (isolated to `archive.py`).

# 🦤 TODO — Exoplanet Transit Lightcurve Reduction

Prioritized roadmap for the planned refactor (Part I) and the web data-input
application (Part II). Every task is traced to its source IDs in
[`REQUIREMENTS.md`](REQUIREMENTS.md) (`R-*` / `W-*` / `NFR-*`),
[`ADR.md`](ADR.md) (`ADR-*`), and [`SPECS.md`](SPECS.md) (`S-*`).

**Priorities:** **P0** blocking/correctness · **P1** important · **P2** nice-to-have
· **P3** docs/polish.

**Acceptance invariant (Part I gate).** Every Part I change MUST keep the *standard*
(dark-subtraction) reduction of WASP-52 b reproducing catalogue values
(R_p ≈ 1.15 R_Jup, ρ ≈ 400 kg/m³, i ≈ 87.3°) within uncertainty
(R-20/R-21, S-26). Ask of any change: *does the standard branch still reproduce
catalogue values?*

**Branch rule.** Work proceeds only on `claude/repo-overview-documentation-brui3h`
— commit and push there, never to `master` (see [`CLAUDE.md`](../../CLAUDE.md)).

> **Strategy.** The refactor builds a **new `exotransit/` package** (S-1) alongside
> the legacy `src/exoplanet_lightcurve/` code, which stays **untouched** until the
> new package reproduces the acceptance invariant — then it is removed in a
> dedicated commit. Legacy correctness bugs are therefore not patched in place;
> they are folded into the new modules and pinned by the regression fixture.

---

## Part I — Pipeline refactor

Grouped for top-down execution: foundation first, then the testing gate, then the
science/correctness pass.

### P0 — Foundation (unblocks everything)

- [x] Scaffold the installable `exotransit/` package per the S-1 layout
      (`config`, `io_fits`, `calibration`, `tracking`, `photometry`, `timebase`,
      `lightcurve`, `planet`, `outputs`, `plots`, `pipeline`, `cli`). Importing any
      module MUST NOT read files, create dirs, write output, or configure logging —
      all execution starts from `pipeline.run()` / the CLI (R-1, R-2, S-1).
- [x] Add `pyproject.toml` (PEP 621) + committed `uv.lock`, `requires-python
      >=3.13`, constrained deps (`numpy`, `matplotlib`, `astropy`, `photutils`,
      `scikit-image`, `tomli-w`), a `[dependency-groups]` dev group
      (`pytest`, `ruff`, `mypy`), and the `exotransit` console entry point; keep
      `pip install .` working (R-4, R-5, S-3, ADR-0002).
- [x] Port the legacy ~2018 API to current `photutils`/`astropy`: `iters=` →
      `maxiters=`; `.area()` method → `.area` property; `DAOStarFinder(...)(image)`
      call pattern; corrected import paths; `np.transpose([x, y])` position arrays;
      explicit masked-array (`.filled(np.nan)`) policy. **Must reproduce the
      acceptance invariant — this is the port gate** (R-10, S-10).
- [x] Externalise config to **TOML** (one file per target): full schema, defaults,
      and `config.load(path) -> Config` validation that fails fast with a
      `ConfigError` naming the offending field/value/expectation. Collapse the five
      mutually-exclusive reduction flags into `reduction.method` + `reduction.cut`
      so illegal combinations are unrepresentable. Ship
      `configs/{wasp52b,hatp19b,tres5b}.toml` from the commented blocks in
      `exo_input_values.py` (R-6, R-7, R-8, R-15, S-4, S-5, ADR-0001).
- [x] Detection guard: select the detected source nearest the window centre; flag
      `NO_SOURCE` / `OFF_CENTER` / `AMBIGUOUS` (and `SATURATED` /
      `REGISTRATION_FAILED`) with per-frame quality flags instead of silently
      indexing `sources[0]`; abort with exit 4 if > 50 % of science frames are
      flagged (R-11, S-11, S-13).
- [x] Validate inputs at discovery (`io_fits.discover`): required dirs exist and are
      non-empty (**no** auto-creation of input dirs — only `paths.output` is
      created), frames within a category share dimensions, required headers
      (`DATE-OBS`/`TIME-OBS`, `EXPTIME`) present; fail with a `DataError` listing
      *every* offending file. Absorbs the legacy empty-dir / `np.stack([])` crash
      (R-18, S-9).

### P0 — Testing gate

- [x] `pytest` unit tests per the S-25 table: `config` (each validation rule),
      `calibration` (combine + reduction arithmetic + flat normalisation),
      `photometry` (hand-computed sky subtraction + detection-guard paths),
      `tracking` (recover injected shift < 0.1 px), `timebase` (BJD_TDB < 1 s vs
      reference), `lightcurve` (N=1 passthrough + ensemble rejection), `planet`
      (R_p/ρ/i + error propagation), `outputs` (CSV/JSON round-trip, config hash)
      (R-20, S-25, ADR-0008).

### P1 — Correctness & science

- [x] Replace module-level globals (`images_array`, `dark_master`, `bias_master`,
      mutating crop coords) with explicit args/returns and frozen dataclasses;
      `measure_star(frames, masters, star, shifts, cfg)` takes everything as
      parameters (R-3, S-2). *(done — `photometry.measure_star` has that exact
      signature; every stage passes state explicitly, no module globals.)*
- [x] Master frame construction honouring `master_dark_combine` /
      `master_bias_combine` — fixes the legacy always-mean bias bug; build only the
      masters the method needs (R-10, S-7). *(done — `calibration.build_masters`
      + `METHOD_NEEDS`; `build_master` honours the per-master combine mode.)*
- [x] Frame calibration ordering (subtract masters per method → optional flat-field
      → per-frame `cut`), with flat-fielding behind an optional `paths.flats` switch
      (R-14, S-8). *(done — `io_fits` discovers `.FLAT.` frames when `paths.flats`
      is set; `calibration.build_master_flat` + `apply_flat`; `measure_star` applies
      subtract → flat → crop → cut. Flats absent = no-op, invariant untouched.)*
- [x] Compute **BJD_TDB** per frame from `DATE-OBS`/`TIME-OBS` + `EXPTIME`
      (mid-exposure) and target/site coordinates; use it as the light-curve time
      ordinate everywhere; retain the raw header string for provenance
      (R-12, S-14, ADR-0003). *(computed in `io_fits.discover`, carried on
      `FrameMeta.bjd_tdb`, written to the CSV; plots/time-windows consume it when
      those P1 items land.)*
- [x] Derive transit windows from predicted ingress/egress times and add a **linear
      baseline fit** for the out-of-transit airmass slope (HAT-P-19 b); retire the
      fixed "first 20 / middle 20 / last 20" windows and `x_ax_loc` tick indices
      (R-13, S-16, S-23). *(done — `planet._linear_depth` fits `a+b·t` over BJD_TDB
      pre/post windows; `timebase.transit_bounds_bjd` converts predicted times;
      `pipeline` dispatches on `baseline_fit`. The fixed 20-frame windows survive
      only in the `"median"` path that pins the invariant; `x_ax_loc` is moot until
      `plots.py` lands.)*
- [x] Derived parameters `R_p` / ρ / `i` (+ max inclination) with Gaussian error
      propagation; document the exact SI inclination formula and its derivation in
      the `planet.py` docstring (R-13, R-25, S-17). *(done — `planet.compute`
      returns all params + errors; the SI inclination relation and its `1e10`
      scale are documented in the module docstring.)*
- [x] Weighted-ensemble differential comparison (Broeg-style): artificial comparison
      star, baseline-only weights, iterative rejection, guarded N=1 passthrough;
      keep per-calibrator diagnostic ratios and record used/rejected membership
      (R-15, S-15, ADR-0005). *(done — `lightcurve._broeg`: own-variance init then
      leave-one-out `1/σ²` weights over the out-of-transit window, gross-outlier
      rejection (5× median σ); N=1 / no-window falls back to equal-weight
      passthrough so the median-baseline invariant is unchanged. Membership +
      weights recorded in the CSV (`in_ensemble_*`) and JSON `ensemble` block.
      Linear+ensemble on the WASP-52 b set → R_p≈1.14, ρ≈417, i≈87.4.)*
- [x] Automatic drift tracking via whole-frame **phase cross-correlation**
      (`skimage.registration`), one global `(dx, dy)` per frame + per-star centroid
      refinement; keep manual `manual_shifts` as an explicit fallback
      (R-16, S-12, ADR-0004). *(done — `tracking.auto_shifts` registers each light
      vs `reference_frame` via `phase_cross_correlation`; per-star centroiding stays
      in photometry; `pipeline.run` dispatches on `mode`; manual/off unchanged.
      Test `test_auto_shift_recovers_injected_offset` recovers an injected drift
      < 0.1 px.)*
- [x] CLI: `exotransit reduce CONFIG.toml` with overrides
      (`--reduction`/`--aperture`/`--output-dir`/`--log-level`/`--no-figures`/
      `--dry-run`) and `exotransit validate CONFIG.toml`; defined exit codes
      (0/2/3/4/1) (R-9, S-6). *(done — `cli.main`; all overrides + exit codes
      0/1/2/3/4 present.)*
- [x] Structured `logging` on the `exotransit.*` hierarchy replacing `print`s
      (including the broken `logger.info('Min:', value)` positional-arg calls);
      handlers configured only in `cli.main()` / `pipeline.run()`; level from
      config, overridable by `--log-level` (R-17, S-24). *(done — module loggers on
      `exotransit.*`; `_configure_logging` in `cli.main`; only CLI result lines
      remain as stdout `print`s.)*
- [x] Machine-readable outputs: extended per-frame **CSV** (BJD_TDB, quality flags,
      ensemble membership, ratios) + **JSON sidecar** (provenance, config SHA-256,
      input file list, method, derived params with uncertainties) — together
      sufficient to regenerate any figure (R-23, R-24, S-21, S-22, ADR-0006).
      *(done — `outputs.write_csv` / `write_json`; verified in [`RUN_REPORT.md`](RUN_REPORT.md) §4.)*
- [x] Make figure generation optional (`output.figures`) and non-fatal — a plotting
      exception logs a warning and never aborts a run after fluxes are computed
      (R-19, S-23). *(done — `pipeline.run` guards `plots.render`; `output.figures`
      gate defaults off.)*
- [x] CI (GitHub Actions) on push + PR: `uv sync` → `ruff check` + `ruff format
      --check` → `mypy exotransit/` (non-strict) → `pytest`, on a Python 3.13 +
      latest matrix (R-22, NFR-2, S-27, ADR-0007). *(done — `.github/workflows/ci.yml`;
      acceptance test self-skips without the WASP-52 b data.)*
- [x] NumPy-style docstrings on all public functions; type hints passing `mypy`
      non-strict; update `README.md` with install + run instructions once packaging
      lands (R-25, NFR-2, S-28). *(done — `mypy exotransit/` clean; `README.md` has
      install/run/physics; public functions carry docstrings.)*

### P2 — After the invariant holds

- [x] Remove the legacy `src/exoplanet_lightcurve/` code in a dedicated commit once
      the new package demonstrates the acceptance invariant (S-1). *(done — all P1
      items landed and `tests/test_acceptance.py` pins the invariant, so the legacy
      `ExoplanetLightcurve.py` / `exo_input_values.py` were `git rm`'d; ruff no longer
      needs to exclude `src/`. Historical `ExoplanetLightcurve.py:NNN` provenance
      citations remain in the `exotransit` source, preserved via git history.)*
- [x] ~~Record full provenance so a run is self-describing (software version, config,
      input file list)~~ (R-24, S-22). *Done — `outputs.write_json` (config SHA-256 +
      input file list) is wired in `pipeline.run()`; verified in [`RUN_REPORT.md`](RUN_REPORT.md) §4.*

---

## Part II — Web data-input application

Browser app that turns "directory of FITS frames → validated pipeline config →
(optional) run" into clicks and validated forms. Thin **FastAPI** backend reusing
the Part I `exotransit` package (no second science implementation), **React** SPA
with a server-tiled canvas viewer, guided-wizard interaction model
(S-29, ADR-0009/0010/0011). Scheduled by milestone.

### M1 — Config generator (P0/P1)

- [x] Session create from uploaded frames or server-side dirs; per-category header
      summary (count, dims, exposure, `TIME-OBS` range, mismatch flags) reusing S-9
      validation (W-1, W-2, S-30). *(done — `webapp/server/sessions.py` +
      `io_fits.summarize` (non-raising, reuses the S-9 helpers; `discover` stays
      the strict gate); `POST /api/sessions`, `/upload`, `GET /summary`.)*
- [x] Interactive frame viewer (linear/log/zscale scaling, zoom, pan) fed by
      backend-rendered tiles (W-4, S-30, ADR-0010). *(done —
      `webapp/server/rendering.py` renders whole-frame PNGs down-sampled to
      ≤1024 px (astropy scaling + matplotlib encoding, no new imaging dep);
      canvas viewer with wheel-zoom/drag-pan in `webapp/client/src/StarsStep.tsx`.)*
- [x] Click-to-pick science target + N calibrators → pixel coords; labelled,
      listed, removable/renamable (W-5, S-30). *(done — click maps display →
      array coords client-side (`star.x` = row, `star.y` = col, legacy
      indexing); science first, then calibrators; list with rename/remove.)*
- [x] Validated parameter forms: single reduction-method choice + photometry params,
      live validation (inner < outer annulus, aperture > 0), run/export disabled
      while invalid (W-8, W-9, S-30). *(done — client checks in
      `webapp/client/src/steps.tsx`; authoritative validation round-trips the
      state through `exotransit.config.load` via `POST /validate`.)*
- [x] Timeline scrubber under the viewer (frame index + timestamp, keyboard
      stepping, pre-cached/down-sampled frames) (W-14, W-NFR-2, S-30). *(done —
      native range input (keyboard for free) + `TIME-OBS` stamp; ±3-frame PNG
      prefetch. Play/pause auto-advance deferred to M2 polish.)*
- [x] Guided wizard (open → check → pick → options → review → export) with
      step-gating and a review screen; export a re-loadable pipeline **TOML**
      (W-11, W-15, W-16, W-19, W-21, W-NFR-1, S-30). *(done —
      `webapp/client/src/App.tsx`; Next disabled with a plain-language reason;
      free revisits keep later values; S-30 gate verified: the web-exported
      WASP-52 b TOML runs unmodified through `exotransit reduce` and reproduces
      the frozen invariant R_p=1.2198 R_Jup, ρ=336.1 kg/m³, i=87.254°.)*

### M2 — Guided photometry (P1)

- [ ] Click-to-centroid snap with aperture + annulus overlay at configured radii
      (W-6, S-31).
- [ ] Drift assistance: mark frame index + new position to build `manual_shifts`
      visually, or preview auto-tracking shifts (W-7, S-31).
- [ ] System-parameter form with NASA Exoplanet Archive lookup by target name
      (manual override always available; lookup failures degrade gently)
      (W-10, W-20, S-31).
- [ ] Freely revisit completed steps without losing later values (W-17, S-31).

### M3 — Run & results (P1/P2)

- [ ] `POST /run` background job + progress; results view with differential light
      curve(s) and derived parameters with uncertainties (W-12, S-32).
- [ ] Live preview: parameter changes after a first result trigger a down-sampled
      re-preview (overlay and/or curve); final saved/run result uses full resolution
      (W-22, W-NFR-2, W-NFR-3, S-32).
- [ ] Optional side-by-side comparison of two reduction methods + CSV download
      (replaces the legacy `option_compare` workflow) (W-13, S-32).

### Cross-cutting (P1/P2)

- [x] Session persistence / resume-later; independent, isolated sessions
      (W-3, W-18, S-29). *(done for M1 scope — one JSON-backed directory per
      session under `_websessions/`, state saved on every step transition,
      resume via the `#sid` URL hash.)*
- [x] Local-first, single-user, no-auth deployment (`uvicorn` + browser); backend
      reuses the Part I package directly (W-NFR-4, W-NFR-5, S-29). *(done —
      `uv run --group web uvicorn webapp.server.main:app` serves API + built SPA
      from `webapp/client/dist`; no auth, no second science implementation.)*

---

## Decisions needed (proposed — blocked on maintainer)

The four additions surfaced in [`SPECS.md`](SPECS.md) §13. **Not scheduled** until
promoted into `REQUIREMENTS.md` as real `R-*` entries; the specs (S-18–S-20, S-4/S-22)
are written so they can be dropped cleanly if rejected.

- [ ] **R-27 (P1?)** — Depth uncertainty from photometric scatter: fold the measured
      baseline/in-transit light-curve scatter into the reported `R_p`/ρ/`i`
      uncertainties, alongside the catalogue errors (S-18, §13.1).
- [ ] **R-28 (P2?)** — T₀ and O−C output: estimate mid-transit time (BJD_TDB) and,
      given a reference ephemeris, report O−C in the results JSON (S-19, §13.2).
- [ ] **R-29 (P2?)** — Limb-darkening caveat + optional `exotransit fit` transit
      model (batman/pytransit) reported next to — not instead of — the √depth
      estimator (S-20, §13.3).
- [ ] **R-30 (P1?)** — `schema_version` field in the TOML config and results JSON so
      future format changes are detectable/migratable (S-4, S-22, §13.4).

---

## Docs & housekeeping (P3)

- [x] Refresh `CLAUDE.md` `file:line` anchors to symbol references once the code is
      modularised (R-26, S-1). *(done alongside the legacy removal — §1–§6 rewritten
      to the `exotransit` package; §2 anchors now name `planet.compute` etc.)*
- [x] Add a `.gitignore` for generated images, logs, and CSV output. *(done —
      `__pycache__`, `*.csv`, `exo_console.log`/`exotransit.log`, and the `_*b/` data
      dirs (which hold `images/`) are ignored; the shipped `configs/*.toml` are now
      tracked, not ignored.)*
- [ ] Document the reduction physics (planet radius / density / inclination) and the
      pipeline stages end to end in `README.md` / a project wiki once modules land
      (R-25, S-17).


---

### Bugfixes & needed Features

- [x] ~~laggy display of images. render them with lower resolution - calculations MUST be done on full and raw FITS image.~~
      *(done differently — HTTP `Cache-Control: max-age=300` on `/frames/{i}/png` +
      widened `render_png` lru_cache 64→256. Resolution left unchanged: integer
      decimation is a no-op at the shipped 1124×850 frame size (562px at both 1024
      and 640 caps), and photometry already reads full-res FITS per W-NFR-3. The lag
      was server re-render on cache miss + browser re-fetch, not pixel count.)*
- [x] ~~Transitduration in "5 System" should be calculated by predicted ingress / egress~~
      *(done — `SystemStep` auto-fills `transit_duration` from the ingress/egress window
      (`windowMinutes` in `steps.tsx`) but keeps it editable, with a `↻ from window` resync
      button. WASP-52b window = 108 min vs catalogue 110, so the override stays available.)*
- [ ] by given a Target name there should be a button that allows to query from a webpage the needed parameters for Stellar Radius, etc
- [x] ~~4 - Parameters: Aperture Radius, Annulus, FWHM, half-width should be displayed live in the "3 - STARS" section (also editable), as you dont see how large the area is.~~
      *(done — `StarsStep` draws aperture (solid) / annulus (dashed) / crop-window (dotted)
      overlays around each picked star, and edits them via a compact field row (shared
      `state.photometry` / `state.stars`, so the Parameters step stays in sync). FWHM is an
      editable field only — it is a detection scale, not an aperture area.)*
- [ ] 4 - Parameters: For selected Science Target and Calibrators display the Integrated Flux by diameter (so x-axis is diameter of the selection, y is the total sum of light that occurs at this radius). Display this as subplots besides the real image.
- [x] ~~4 - Parameters: Counting of Reference Frame should start with 1 (and also select the first image then), same as in the "3- Stars" timeline scrubber.~~
      *(done — the Reference frame field now displays 1-based to match the Stars scrubber;
      storage stays 0-based as the array index (`tracking.py` clamps it). Default 0 shows "1"
      = the first image.)*
- [ ] add section "7 - PREVIEW" which shows a plot of the tracked target (like in "3-Stars") and the calculated photometric value in a subplot below, so the lightcurve can easily be seen
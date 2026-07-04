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

- [ ] Replace module-level globals (`images_array`, `dark_master`, `bias_master`,
      mutating crop coords) with explicit args/returns and frozen dataclasses;
      `measure_star(frames, masters, star, shifts, cfg)` takes everything as
      parameters (R-3, S-2).
- [ ] Master frame construction honouring `master_dark_combine` /
      `master_bias_combine` — fixes the legacy always-mean bias bug; build only the
      masters the method needs (R-10, S-7).
- [ ] Frame calibration ordering (subtract masters per method → optional flat-field
      → per-frame `cut`), with flat-fielding behind an optional `paths.flats` switch
      (R-14, S-8).
- [x] Compute **BJD_TDB** per frame from `DATE-OBS`/`TIME-OBS` + `EXPTIME`
      (mid-exposure) and target/site coordinates; use it as the light-curve time
      ordinate everywhere; retain the raw header string for provenance
      (R-12, S-14, ADR-0003). *(computed in `io_fits.discover`, carried on
      `FrameMeta.bjd_tdb`, written to the CSV; plots/time-windows consume it when
      those P1 items land.)*
- [ ] Derive transit windows from predicted ingress/egress times and add a **linear
      baseline fit** for the out-of-transit airmass slope (HAT-P-19 b); retire the
      fixed "first 20 / middle 20 / last 20" windows and `x_ax_loc` tick indices
      (R-13, S-16, S-23).
- [ ] Derived parameters `R_p` / ρ / `i` (+ max inclination) with Gaussian error
      propagation; document the exact SI inclination formula and its derivation in
      the `planet.py` docstring (R-13, R-25, S-17).
- [ ] Weighted-ensemble differential comparison (Broeg-style): artificial comparison
      star, baseline-only weights, iterative rejection, guarded N=1 passthrough;
      keep per-calibrator diagnostic ratios and record used/rejected membership
      (R-15, S-15, ADR-0005).
- [ ] Automatic drift tracking via whole-frame **phase cross-correlation**
      (`skimage.registration`), one global `(dx, dy)` per frame + per-star centroid
      refinement; keep manual `manual_shifts` as an explicit fallback
      (R-16, S-12, ADR-0004).
- [ ] CLI: `exotransit reduce CONFIG.toml` with overrides
      (`--reduction`/`--aperture`/`--output-dir`/`--log-level`/`--no-figures`/
      `--dry-run`) and `exotransit validate CONFIG.toml`; defined exit codes
      (0/2/3/4/1) (R-9, S-6).
- [ ] Structured `logging` on the `exotransit.*` hierarchy replacing `print`s
      (including the broken `logger.info('Min:', value)` positional-arg calls);
      handlers configured only in `cli.main()` / `pipeline.run()`; level from
      config, overridable by `--log-level` (R-17, S-24).
- [ ] Machine-readable outputs: extended per-frame **CSV** (BJD_TDB, quality flags,
      ensemble membership, ratios) + **JSON sidecar** (provenance, config SHA-256,
      input file list, method, derived params with uncertainties) — together
      sufficient to regenerate any figure (R-23, R-24, S-21, S-22, ADR-0006).
- [ ] Make figure generation optional (`output.figures`) and non-fatal — a plotting
      exception logs a warning and never aborts a run after fluxes are computed
      (R-19, S-23).
- [ ] CI (GitHub Actions) on push + PR: `uv sync` → `ruff check` + `ruff format
      --check` → `mypy exotransit/` (non-strict) → `pytest`, on a Python 3.13 +
      latest matrix (R-22, NFR-2, S-27, ADR-0007).
- [ ] NumPy-style docstrings on all public functions; type hints passing `mypy`
      non-strict; update `README.md` with install + run instructions once packaging
      lands (R-25, NFR-2, S-28).

### P2 — After the invariant holds

- [ ] Remove the legacy `src/exoplanet_lightcurve/` code in a dedicated commit once
      the new package demonstrates the acceptance invariant (S-1). *Deferred: the
      invariant is demonstrated locally (WASP-52 b, see [`RUN_REPORT.md`](RUN_REPORT.md)),
      but legacy stays the reference until the open P1 correctness items land.*
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

- [ ] Session create from uploaded frames or server-side dirs; per-category header
      summary (count, dims, exposure, `TIME-OBS` range, mismatch flags) reusing S-9
      validation (W-1, W-2, S-30).
- [ ] Interactive frame viewer (linear/log/zscale scaling, zoom, pan) fed by
      backend-rendered tiles (W-4, S-30, ADR-0010).
- [ ] Click-to-pick science target + N calibrators → pixel coords; labelled,
      listed, removable/renamable (W-5, S-30).
- [ ] Validated parameter forms: single reduction-method choice + photometry params,
      live validation (inner < outer annulus, aperture > 0), run/export disabled
      while invalid (W-8, W-9, S-30).
- [ ] Timeline scrubber under the viewer (frame index + timestamp, keyboard
      stepping, pre-cached/down-sampled frames) (W-14, W-NFR-2, S-30).
- [ ] Guided wizard (open → check → pick → options → review → export) with
      step-gating and a review screen; export a re-loadable pipeline **TOML**
      (W-11, W-15, W-16, W-19, W-21, W-NFR-1, S-30).

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

- [ ] Session persistence / resume-later; independent, isolated sessions
      (W-3, W-18, S-29).
- [ ] Local-first, single-user, no-auth deployment (`uvicorn` + browser); backend
      reuses the Part I package directly (W-NFR-4, W-NFR-5, S-29).

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

- [ ] Refresh `CLAUDE.md` `file:line` anchors to symbol references once the code is
      modularised (R-26, S-1).
- [ ] Add a `.gitignore` for generated images, logs, and CSV output.
- [ ] Document the reduction physics (planet radius / density / inclination) and the
      pipeline stages end to end in `README.md` / a project wiki once modules land
      (R-25, S-17).

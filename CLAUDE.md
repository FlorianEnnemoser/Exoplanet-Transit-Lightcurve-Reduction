# CLAUDE.md

Guidance for agents in this repo, in a scientific register: what the pipeline does, how each stage maps to the code, and the caveats needed to operate it.

> **Active development branch.** Work proceeds **only** on `develop` — commit and push there, **never to `master`**. **Do not open or ask about pull requests**; the maintainer merges manually. See [`.claude/docs/REQUIREMENTS.md`](.claude/docs/REQUIREMENTS.md) for the planned refactor and the web data-input application.

## Documentation map

- [`README.md`](README.md) — project overview + install/run instructions.
- [`LICENSE`](LICENSE) — GPL v3 full text.
- [`.claude/docs/REQUIREMENTS.md`](.claude/docs/REQUIREMENTS.md) — engineering requirements (R-*/W-* IDs) for the planned refactor and web data-input app.
- [`.claude/docs/ADR.md`](.claude/docs/ADR.md) — architecture decision records (MADR-style); ADR-0001: config format = TOML.
- [`.claude/docs/SPECS.md`](.claude/docs/SPECS.md) — engineering specs (S-* IDs) implementing REQUIREMENTS.md + ADR.md.
- [`.claude/docs/TODO.md`](.claude/docs/TODO.md) — prioritized roadmap (P0/P1/…).

## Provenance

Analysis pipeline of the bachelor thesis *"Photometrische Messungen von Exoplanetentransits — Einfluss der Reduktionsmethode auf die Ergebnisse"* (F. Ennemoser; supervisor Dr. T. Ratzka; Karl-Franzens-Universität Graz, 2018). Its question is **methodological** — *how does the choice of CCD reduction method change the derived planet parameters?* — so the code is a switchboard of reduction paths (§4) reused across targets. Built on `astropy` examples + supervisor IDL code; `aperture_photometry` ≡ NASA IDL `APER`. Licence: **GPL v3**.

The original single-file thesis script (`src/exoplanet_lightcurve/ExoplanetLightcurve.py` + `exo_input_values.py`) has been **refactored into the installable `exotransit` package and removed**; its numerics are folded into the new modules and pinned by the acceptance regression test (`tests/test_acceptance.py`). Code comments still cite `ExoplanetLightcurve.py:NNN` as historical provenance for each ported formula — those anchors point at the removed file, preserved in git history.

## Coding Standards and Precedures

- always ask if something is unclear
- never use `from __future__ import annotations`
- always use f-strings (e.g.: `f"{mydata}"`) for logging or printing
- always put imports on top of the file (not inside functions)
- keep `.py` files under 600 lines.
- after implementing a TODO, cross it out in `.claude/docs/TODO.md`
- after implementing, provide a copyable commit message with a Header, short description and Changelog: <trailer> (gitlab style).

## 1. Overview

A ground-based **differential aperture-photometry** pipeline for transiting hot Jupiters: from a CCD time series it extracts the flux of one science target and N ≥ 1 calibrators, forms a differential light curve to cancel atmospheric/instrumental systematics, measures the transit depth, and derives planet radius, density, and inclination with Gaussian-propagated errors. Data: Lustbühel Observatory (Graz), ASA 500 mm f/9 Cassegrain + SBIG STF-8300 CCD, 3×3 binning, Sloan r′, 2016. **Targets:** WASP-52 b (active), HAT-P-19 b, TrES-5 b succeeded; KELT-16 b (out-of-focus frames → detection failed) and TrES-2 b (not 3×3 binned) failed. Packaged as the installable **`exotransit`** module set (import-safe; all execution starts from `pipeline.run()` / the `exotransit` CLI), TOML-configured, with a `pytest` suite and GitHub Actions CI.

## 2. Key relations

Derived-parameter arithmetic lives in [`exotransit/planet.py`](exotransit/planet.py) `compute()` (SI relations + Gaussian error propagation documented in the module docstring):

- Transit depth `ΔF/F ≈ (R_p/R_*)²`; `R_p = √((ΔF/F)·R_*²)`
- Bulk density `ρ_p = 3M_p/(4πR_p³)` (planet mass from NASA Exoplanet Archive)
- Inclination by inverting `t_ges ≈ (P/π)·arcsin√(R_*²/a² − cos²i)`
- Calibration `Light_red = Light − Dark` (standard) in [`calibration.subtract_masters`](exotransit/calibration.py); optional flat-fielding in `calibration.apply_flat` (behind `paths.flats`). Differential ratios `F_sci/F_cal_k` and the weighted ensemble `F_sci/C` (≈1 out of transit, <1 in transit) in [`lightcurve.differential`](exotransit/lightcurve.py).

## 3. Repository layout

`exotransit/` — the installable package: `config` (TOML load + validation), `io_fits` (FITS discovery/validation, BJD_TDB wiring), `calibration` (masters, flat, per-frame reduction), `tracking` (auto phase-correlation + manual shifts), `photometry` (detection guard + aperture photometry), `timebase` (BJD_TDB, transit-window bounds), `lightcurve` (differential ratios + Broeg ensemble), `planet` (depth → R_p/ρ/i), `outputs` (CSV + JSON sidecar), `plots` (figures, optional), `pipeline` (`run()` orchestration), `cli` (`exotransit` entry point). · `configs/{wasp52b,hatp19b,tres5b}.toml` (one committed config per target) · `webapp/` — Part II data-input app: `server/` (FastAPI wrapping `exotransit`; `uv run --group web uvicorn webapp.server.main:app`) + `client/` (Vite/React wizard SPA, NASA-styled per SPECS S-33; `npm run dev`, or build and let the backend serve `dist/`); sessions in gitignored `_websessions/` · `tests/` (`pytest` suite incl. `test_acceptance.py` invariant) · `pyproject.toml` (`uv`-managed packaging/deps + ruff/mypy config) · `.github/workflows/ci.yml` · `README.md` · `LICENSE` (GPL v3) · `CLAUDE.md` · `.claude/docs/{REQUIREMENTS,ADR,SPECS,TODO}.md` — see Documentation map. Data dirs are **not** versioned; only `paths.output` is created on demand. WASP-52 b ships flat in `_WASP52b/` with suffix naming (`*.BIAS.FIT` / `*.DARK.FIT` / rest = lights); outputs to `_WASP52b/images/`.

## 4. Reduction methods & finding

The five mutually-exclusive legacy flags collapsed into `reduction.method` (master subtraction) + an optional `reduction.cut` (per-frame pedestal), so illegal combinations are unrepresentable ([`config.py`](exotransit/config.py); [`calibration.py`](exotransit/calibration.py)):

| `reduction.method` | Operation |
|--------|-----------|
| `none` | raw lights |
| **`standard`** (default) | `Light − MasterDark` |
| `bias` | `Light − MasterBias` |
| `dark_bias` | `Light − MasterBias − MasterDark` |

`reduction.cut` ∈ `none` / `median` / `average` / `min` / `sigma_clip` applies an optional per-frame pedestal subtraction on the cropped window (the legacy `bias_min` = `bias` + `cut="min"`, `bias_sigma` = `bias` + `cut="sigma_clip"`). Optional flat-fielding switches on when `paths.flats` is set. **Finding:** the *standard (dark)* reduction is most accurate (matches NASA Archive/ETD: WASP-52 b R_p≈1.15 R_Jup, ρ≈400 kg/m³, i≈87.3°); **bias+median inflated R_p by >0.1 R_Jup and is unusable**. Acceptance gate for any change: *does the standard branch still reproduce catalogue values?* — enforced by `tests/test_acceptance.py`.

## 5. Configuration & run

One **TOML** config per target ([`configs/*.toml`](configs)), loaded and validated by [`config.load`](exotransit/config.py) (fails fast with a `ConfigError` naming the offending field). Sections: `[observation]`, `[paths]` (lights/darks/bias/flats/output), `[stars]` (science + N calibrators + `crop_half_width`), `[reduction]` (§4), `[photometry]`, `[tracking]` (auto/manual/off), `[transit]` (predicted ingress/egress + `baseline_fit`), `[system]` (catalogue params + target RA/Dec + observatory site for BJD_TDB), `[output]`, `[logging]`. **Switch target** = pick a different config file. **Run** (deps pinned in `pyproject.toml` + `uv.lock`, `requires-python >=3.13`):

```bash
uv run exotransit validate configs/wasp52b.toml   # parse + validate only
uv run exotransit reduce   configs/wasp52b.toml    # full reduction
```

Overrides: `--reduction --aperture --output-dir --log-level --no-figures --dry-run`. Writes `<casename>_lightcurve.csv` + `<casename>_result.json` to `paths.output`, logs on the `exotransit.*` logger, and prints R_p/ρ/i. Exit codes: 0 ok · 2 config · 3 data · 4 reduction · 1 unexpected.

## 6. Caveats

- **Acceptance invariant is the gate**: the standard + `baseline_fit="median"` + sci/cal1 depth path for WASP-52 b must keep reproducing catalogue values (`tests/test_acceptance.py`, frozen at R_p=1.2198 R_Jup, ρ=336.1 kg/m³, i=87.254°). The equal-weight ensemble / median path is bit-for-bit; the Broeg ensemble is the depth source only for `baseline_fit="linear"`.
- **Manual drift tracking remains a limitation** where auto registration is off: crop windows shifted by hand via `[tracking].manual_shifts`, re-tuned per dataset. KELT-16 b failed when out-of-focus early frames made `DAOStarFinder` lock onto background. Auto phase-cross-correlation (`tracking.mode = "auto"`) is the default alternative.
- **Airmass baseline slope** (HAT-P-19 b): the `linear` baseline fits `a + b·t` over the out-of-transit windows and evaluates at mid-transit (`planet._linear_depth`), rather than a single endpoint.
- **No flat-fielding data ships**: the flat path (`paths.flats`) is implemented and unit-tested but exercised only synthetically — no real flat frames exist. First-order results (√depth estimator ignores limb darkening; see the R-27/R-29 deferrals in `.claude/docs/TODO.md`).

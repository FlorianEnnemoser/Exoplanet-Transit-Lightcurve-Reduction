# CLAUDE.md

Guidance for agents in this repo, in a scientific register: what the pipeline does, how each stage maps to the code, and the caveats needed to operate it.

> **Active development branch.** Work proceeds **only** on `develop` — commit and push there, **never to `master`**. **Do not open or ask about pull requests**; the maintainer merges manually. See [`.claude/docs/REQUIREMENTS.md`](.claude/docs/REQUIREMENTS.md) for the planned refactor and the web data-input application.

## Documentation map

- [`README.md`](README.md) — one-line project summary.
- [`LICENSE`](LICENSE) — GPL v3 full text.
- [`.claude/docs/REQUIREMENTS.md`](.claude/docs/REQUIREMENTS.md) — engineering requirements (R-*/W-* IDs) for the planned refactor and web data-input app.
- [`.claude/docs/ADR.md`](.claude/docs/ADR.md) — architecture decision records (MADR-style); ADR-0001: config format = TOML.
- [`.claude/docs/SPECS.md`](.claude/docs/SPECS.md) — engineering specs (S-* IDs) implementing REQUIREMENTS.md + ADR.md.
- [`.claude/docs/TODO.md`](.claude/docs/TODO.md) — prioritized roadmap (P0/P1/…).

## Provenance

Analysis pipeline of the bachelor thesis *"Photometrische Messungen von Exoplanetentransits — Einfluss der Reduktionsmethode auf die Ergebnisse"* (F. Ennemoser; supervisor Dr. T. Ratzka; Karl-Franzens-Universität Graz, 2018). Its question is **methodological** — *how does the choice of CCD reduction method change the derived planet parameters?* — so the code is a switchboard of reduction paths (§5) reused across targets. Built on `astropy` examples + supervisor IDL code; `aperture_photometry` ≡ NASA IDL `APER`. Licence: **GPL v3**.

## Coding Standards and Precedures

- always ask if something is unclear
- never use `from __future__ import annotations`
- keep `.py` files under 600 lines.
- after implementing a TODO, cross it out in `.claude/docs/TODO.md`

## 1. Overview

A ground-based **differential aperture-photometry** pipeline for transiting hot Jupiters: from a CCD time series it extracts the flux of one science target and two calibrators, forms a differential light curve to cancel atmospheric/instrumental systematics, measures the transit depth, and derives planet radius, density, and inclination with Gaussian-propagated errors. Data: Lustbühel Observatory (Graz), ASA 500 mm f/9 Cassegrain + SBIG STF-8300 CCD, 3×3 binning, Sloan r′, 2016. **Targets:** WASP-52 b (active), HAT-P-19 b, TrES-5 b succeeded; KELT-16 b (out-of-focus frames → detection failed) and TrES-2 b (not 3×3 binned) failed. Two modules, packaged for `uv` under `src/exoplanet_lightcurve/`; still no tests.

## 2. Key relations

- Transit depth `ΔF/F ≈ (R_p/R_*)²`; `R_p = √((ΔF/F)·R_*²)` → `src/exoplanet_lightcurve/ExoplanetLightcurve.py:440`
- Bulk density `ρ_p = 3M_p/(4πR_p³)` → `:447` (planet mass from NASA Exoplanet Archive)
- Inclination by inverting `t_ges ≈ (P/π)·arcsin√(R_*²/a² − cos²i)` → `:453`
- Calibration `Light_red = Light − Dark` (standard); flat-fielding is **not** done. Differential curves `F_sci/F_cal1`, `F_sci/F_cal2` (≈1 out of transit, <1 in transit) at `:370`.

## 3. Repository layout

`src/exoplanet_lightcurve/ExoplanetLightcurve.py` (procedural pipeline, single entry point) · `src/exoplanet_lightcurve/exo_input_values.py` (configuration as code) · `pyproject.toml` (`uv`-managed packaging/deps) · `tests/` (placeholder, no tests yet) · `README.md` · `LICENSE` (GPL v3) · `CLAUDE.md` · `.claude/docs/{REQUIREMENTS,ADR,SPECS,TODO}.md` (architecture decisions; config format = TOML per ADR-0001) — see Documentation map above. Data dirs are not versioned; created on demand from config paths. WASP-52 b layout: `_WASP52b/{WASP52b,Dark,Bias}/` (inputs), `_WASP52b/images/` (PNG outputs).

## 4. Pipeline stages (`uv run python -m exoplanet_lightcurve.ExoplanetLightcurve`)

No `__main__` guard — **importing runs the full reduction**. Stages:

1. Logger init (`:21`); missing data dirs auto-created.
2. **Master dark** — mean (`combo_master_dark==1`) or median (`==2`) combine (`:67`); **master bias** — mean (`:99`). Saved as PNGs.
3. **`fluxtarget()`** (`:117`), per star (science + 2 calibrators). Per frame: calibrate per selected mode (`:128`) → crop a square window at the star's DS9-picked coords (`:142`) → optional pedestal/`sigma_clip` (`:184`) → `DAOStarFinder` detection vs `sigma_clipped_stats` background (`:202`) → `CircularAperture` + `CircularAnnulus` photometry (`:212`) → local sky subtraction `residual = aperture_sum − annulus_mean×aperture_area` (`:223`) → **manual drift shift** at frame indices `i` by `shift_x/shift_y` (`:244`). Defaults: aperture 4 px, annulus 6/8 px (≈3·FWHM ⇒ ~90% of light).
4. **Differential curve** (`:319`+): read `TIME-OBS` header (`:350`), optional CSV (`:357`), form ratios (`:370`).
5. **Transit depth** (`:386`): median of first/last 20 (baseline) vs central 20 (in-transit); depth = out-of-transit baseline − in-transit median.
6. **Planet parameters** (`:431`): depth→mag, `R_p` (`:440`), density (`:447`), inclination + max inclination (`:453`), printed to stdout.
7. **Plots** (`:469`+): `sci-cal1`, `sci-cal2`, optional delta plots, transit markers.

## 5. Reduction methods (config flags) & finding

| Method | Flag(s) | Operation |
|--------|---------|-----------|
| None | `no_red=1` | raw lights |
| **Standard (default)** | `dark_red=1` | `Light − MasterDark` |
| Bias | `bias_red=1` | `Light − MasterBias` |
| Bias + Median | `bias_red`/`bias_min` + `median_cut=1` | bias, then per-frame median subtraction |
| Reduced dark count | `dark_red=1`, few darks | standard, fewer darks combined |

Flags are **mutually exclusive** — enable exactly one (first matching branch wins). **Finding:** the *standard (dark)* reduction is most accurate (matches NASA Archive/ETD: WASP-52 b R_p≈1.15 R_Jup, ρ≈400 kg/m³, i≈87.3°); **bias+median inflated R_p by >0.1 R_Jup and is unusable**. Acceptance gate for any change: *does the standard branch still reproduce catalogue values?*

## 6. Configuration & run (`exo_input_values.py`)

Configuration *as code*, imported as a namespace. Groups: paths; star names & `[x,y]` coords + `pix_around_star`; `combo_master_*`; reduction flags (§5); detection/photometry (`background_sigma`, `fwhm`, `threshold`, `aperture`, `annulus`, `methods`); manual tracking (`i`, `shift_x`, `shift_y`); transit window; CSV I/O; system params (`rstar`,`e_rstar`,`a`,`P`,`m_planet`,`e_m_planet`,`trandur`); physical constants (below "DO NOT EDIT"). **Switch target** by commenting out the active block and uncommenting another (WASP-52 b active; HAT-P-19 b / TrES-5 b / KELT-16 b provided). **Run:** populate input dirs with FITS frames, edit config, then `uv run python -m exoplanet_lightcurve.ExoplanetLightcurve` → results CSV, PNG figures, `exo_console.log`, params on stdout. Deps pinned in `pyproject.toml`: `numpy`, `matplotlib`, `astropy`, `photutils` (legacy versions, see §7).

## 7. Caveats

- **Runs on import** (no `main()` guard) — never import for introspection; **single active flag** (multiple mutually-exclusive flags → undefined); **fixed 3-star model** hard-wired in MAIN.
- **Manual drift tracking is the principal limitation**: crop windows shifted by hand via `i`/`shift_x`/`shift_y`, re-tuned per dataset. KELT-16 b failed when out-of-focus early frames made `DAOStarFinder` lock onto background.
- **Airmass baseline slope** (HAT-P-19 b): depth uses the in-transit median vs the **mean of pre- and post-transit baselines**, not one endpoint.
- **Legacy ~2018 API**: `iters=` (now `maxiters=`), `aperture.area()` method (now `.area` property), callable `DAOStarFinder` — will not run unmodified; pin an old env or port.
- **No flat-fielding, no tests, no CI.** Dependency manifest exists (`pyproject.toml`) but is unverified against the legacy API. First-order results.

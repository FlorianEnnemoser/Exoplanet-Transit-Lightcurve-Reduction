# Exoplanet-Transit-Lightcurve-Reduction

Ground-based **differential aperture-photometry** pipeline for transiting hot
Jupiters. From a CCD time series it extracts the flux of one science target and
N calibrators, forms a differential light curve to cancel atmospheric and
instrumental systematics, measures the transit depth, and derives the planet
radius, density, and inclination with Gaussian-propagated errors.

Refactor of the bachelor-thesis analysis code (F. Ennemoser, *Photometrische
Messungen von Exoplanetentransits*, Univ. Graz, 2018). Data: Lustbühel
Observatory, 2016. Licence: **GPL v3**.

## Install

Requires Python ≥ 3.13.

```bash
uv sync                 # dev + runtime deps from the committed uv.lock
# or, without uv:
pip install .
# or, with uv as editable:
uv pip install . -e
```

## Run

Each target is one TOML config. Point its `[paths]` at the FITS frames, then:

```bash
uv run exotransit validate configs/wasp52b.toml     # parse + validate only
uv run exotransit reduce   configs/wasp52b.toml     # full reduction
```

`reduce` writes `<casename>_lightcurve.csv` (per-frame flux, quality flags,
ensemble membership, ratios) and `<casename>_result.json` (provenance, config
SHA-256, derived parameters with uncertainties) to `[paths].output`, and prints
R_p, ρ, i to stdout. Useful overrides: `--reduction`, `--aperture`,
`--output-dir`, `--log-level`, `--no-figures`, `--dry-run`.

Exit codes: `0` success · `2` config error · `3` data error · `4` reduction
error (too many frames flagged) · `1` unexpected.

## What it computes

- **Calibration** — master dark/bias per `[reduction].method`
  (`none`/`standard`/`bias`/`dark_bias`), optional flat-fielding, optional
  per-frame pedestal `cut`.
- **Tracking** — auto drift registration (phase cross-correlation) or manual
  step shifts.
- **Timebase** — mid-exposure **BJD_TDB** per frame (target + site corrected).
- **Differential curve** — per-calibrator ratios `F_sci/F_cal_k` and a
  Broeg-style variance-weighted ensemble with iterative rejection.
- **Transit depth** `ΔF/F` from median windows (legacy) or a linear
  baseline fit over BJD_TDB transit windows.
- **Planet parameters** — `R_p = √(ΔF/F)·R_*`, `ρ_p = 3M_p/(4πR_p³)`, and the
  inclination from `cos i = √(((R_*+R_p)/a)² − sin²(π·t_dur/P))`, with Gaussian
  error propagation from the catalogue uncertainties.

The *standard* (dark-subtraction) reduction of WASP-52 b is the acceptance
gate: R_p ≈ 1.15 R_Jup, ρ ≈ 400 kg/m³, i ≈ 87.3°.

## Web app

A browser wizard (`webapp/`) builds a config interactively: FastAPI backend
wrapping `exotransit` + a Vite/React SPA. All commands run from the repo root
unless noted.

### Development (two processes, live reload)

```bash
uv run --group web uvicorn webapp.server.main:app --reload   # API on :8000
cd webapp/client && npm install && npm run dev               # SPA on :5173
```

Open <http://localhost:5173>; Vite proxies `/api` to the backend on `:8000`.
Edits to `webapp/client/src/*` hot-reload instantly — no build step.

### Production (one process)

The backend serves the compiled SPA from `webapp/client/dist/` when present, so
after building you only run uvicorn:

```bash
cd webapp/client && npm install && npm run build   # emits webapp/client/dist/
cd ../..                                            # back to repo root
uv run --group web uvicorn webapp.server.main:app   # serves API + SPA on :8000
```

Open <http://localhost:8000>. **`dist/` is a build artifact**: any change under
`webapp/client/src/` only shows up after re-running `npm run build` (and a
hard browser reload). Use the dev setup above to avoid rebuilding while working.

## Build the package

```bash
uv build            # wheel + sdist into dist/
# or without uv:
python -m build
```

## Develop

```bash
uv run ruff check && uv run ruff format --check
uv run mypy exotransit/
uv run pytest
```

# 🦤 TODO — Exoplanet Transit Lightcurve Reduction

Roadmap of what still needs to be implemented for the pipeline to fulfil its
goal (*"reducing image noise to analyze transiting exoplanet lightcurves"*) as
a working, maintainable project. Tasks are grouped by priority. Line references
point at `ExoplanetLightcurve.py` unless noted otherwise.

---

## P0 — Make it run again (unblock the pipeline)

The script currently cannot run against modern `astropy`/`photutils`.

- [ ] Add a `requirements.txt` (or `pyproject.toml`) pinning `numpy`,
      `matplotlib`, `astropy`, `photutils` to compatible versions.
- [ ] Replace deprecated `iters=` with `maxiters=` in `sigma_clip(...)` and
      `sigma_clipped_stats(...)` (lines 194, 200, 202).
- [ ] Replace `apertures.area()` / `annulus_apertures.area()` method calls with
      the `.area` property (lines 223–224).
- [ ] Fix `photutils` import paths for the current API
      (`DAOStarFinder`, `aperture_photometry`, `CircularAperture`,
      `CircularAnnulus`, `find_peaks`) (lines 12–14).
- [ ] Wrap the executable body in an `if __name__ == '__main__':` guard so the
      module can be imported without running the whole pipeline.

## P1 — Correctness fixes

- [ ] Fix the `logger.info('Min:', np.min(...))`-style calls — pass a single
      formatted string, not extra positional args (lines 154–157).
- [ ] Fix the dead `sigma_red == 1` branch: `sigma_red` is a list `[0,1,1]`, so
      the comparison never matches — test `sigma_red[0] == 1` (line 199;
      config `exo_input_values.py:94`).
- [ ] Honor `combo_master_bias`: the bias master is always averaged regardless
      of the config toggle (line 99; config `exo_input_values.py:61`).
- [ ] Guard the `create_reduced == 1` block against a `NameError` — it uses
      `peaks`, which only exists when `add_local_peaks == 1` (lines 234–255).
- [ ] Handle empty/missing FITS directories with a clear error instead of
      auto-creating an empty dir and crashing on `np.stack([])`
      (lines 38–65, 84–97).
- [ ] Remove the hardcoded ≥20-frames / fixed `x_ax_loc` assumptions in the
      transit-depth estimate so short datasets don't break (lines 386–427,
      482–484).

## P1 — Reduction & photometry completeness

- [ ] Add an optional **flat-field** calibration step (currently only
      dark/bias are handled) to complete standard CCD reduction.
- [ ] Review the master dark/bias construction and background/annulus
      subtraction against the thesis method and document the chosen defaults.

## P2 — Automated star tracking

- [ ] Replace the hardcoded manual `i` / `shift_x` / `shift_y` drift arrays with
      real automated centroid tracking (e.g. centroiding or cross-correlation)
      so field drift is corrected without hand-tuning
      (config `exo_input_values.py:110–115, 157–159`; self-noted TODO in code).

## P2 — Structure & usability

- [ ] Split the monolithic script into modules (io / calibration / photometry /
      lightcurve / plotting) with reusable functions.
- [ ] Add a CLI (argparse) or per-target config file so a new target no longer
      requires editing and un-commenting blocks in `exo_input_values.py`.

## P2 — Validation & tests

- [ ] Add unit tests for the reduction and differential-photometry math
      (master combine, background subtraction, flux ratios, planet parameters).
- [ ] Add a small sample-data smoke test that runs the pipeline end-to-end.
- [ ] Add a `.gitignore` for generated images, logs, and CSV output.

## P3 — Documentation

- [ ] Expand `README.md` with install instructions, expected FITS data layout
      (requires a `TIME-OBS` header), how to configure a target, and the
      produced outputs.
- [ ] Document the physics behind the planet radius / density / inclination
      calculations (lines 431–466).
- [ ] Write a project wiki that describes how this project works end to end —
      the reduction pipeline stages (calibration → photometry → differential
      light curve → planetary parameters), the data flow between them, and the
      role of the key configuration options in `exo_input_values.py`.

## P3 — CI

- [ ] Add a GitHub Actions workflow to run linting and the test suite on push.

# CLAUDE.md

Guidance for AI agents (and human contributors) working in this repository,
written in a scientific register. It describes the instrument-independent
data-reduction methodology implemented here, maps each stage to the source
code, and records the conventions and caveats necessary to operate or extend
the pipeline safely.

> **Provenance note.** This project accompanies a bachelor thesis by
> Florian Ennemoser (Karl-Franzens University Graz, Institute of Geophysics,
> Astrophysics and Meteorology; supervisor T. Ratzka; April 2018). The thesis
> PDF could not be retrieved in this environment (host egress-blocked), so the
> documentation below is derived from the source code and standard
> photometric-reduction theory. Observatory/instrument specifics, target
> selection rationale, and any limb-darkening treatment described in the thesis
> are therefore not reflected here; fold them in if the text becomes available.

---

## 1. Abstract

This repository implements a **ground-based differential aperture-photometry
pipeline** for the detection and characterisation of **transiting extrasolar
planets**. From a time series of calibrated CCD frames it extracts the
instrumental flux of a science target and two comparison ("calibrator") stars,
forms a differential light curve to suppress atmospheric and instrumental
systematics, measures the fractional flux decrement produced by a planetary
transit, and derives first-order physical parameters of the planet — radius,
bulk density, and orbital inclination — with propagated uncertainties. The
implementation comprises two Python modules and is driven entirely by a static
configuration file; there is no packaging, test suite, or dependency manifest.

## 2. Scientific background

**Transit photometry.** When an exoplanet crosses the disc of its host star
along the line of sight, it occults a fraction of the stellar surface, producing
a periodic, box-to-U-shaped dip in the observed flux. To first order the
fractional depth of the dip equals the squared planet-to-star radius ratio,
`ΔF/F ≈ (R_p / R_*)²`, so a measured transit depth combined with an independent
estimate of `R_*` yields the planetary radius `R_p`.

**CCD calibration.** Raw frames carry an additive electronic pedestal (**bias**)
and a thermal signal that accumulates with exposure time (**dark current**).
These are removed by subtracting master calibration frames built by combining
many bias/dark exposures (see §4.2). Flat-fielding (pixel-to-pixel sensitivity
correction) is **not** performed by this pipeline.

**Differential photometry.** Absolute photometry from the ground is corrupted by
time-variable atmospheric transparency and airmass. Dividing the target flux by
the flux of nearby comparison stars of comparable colour cancels these common-mode
signals to first order, leaving a normalised light curve on which the transit
signature is measurable. This pipeline forms `F_target / F_cal1` and
`F_target / F_cal2` as two independent differential light curves.

**Derived quantities.** From the median out-of-transit baseline and the median
in-transit flux the pipeline estimates the transit depth, then computes the planet
radius, bulk density `ρ = 3M_p / (4πR_p³)`, and orbital inclination from the
transit-duration/geometry relation, each with a propagated error.

## 3. Repository structure

```
.
├── ExoplanetLightcurve.py   # Main procedural pipeline (single entry point)
├── exo_input_values.py      # Configuration module (all tunable parameters)
├── README.md                # One-line description
├── LICENSE                  # GNU GPL v3
└── CLAUDE.md                # This document
```

Runtime data directories are **not** version-controlled; they are created on
demand (`os.makedirs`) from the paths in `exo_input_values.py`. For the active
WASP-52 b configuration the expected layout is:

```
_WASP52b/
├── WASP52b/   # science ("light") FITS frames        (input)
├── Dark/      # dark calibration FITS frames          (input)
├── Bias/      # bias calibration FITS frames          (input)
└── images/    # generated PNG figures and graphs      (output)
```

## 4. Methodology — pipeline stages

The program is a top-to-bottom procedural script executed as
`python ExoplanetLightcurve.py`. There is no `if __name__ == "__main__"` guard;
importing the module runs the full reduction. The only reusable routine is
`fluxtarget(...)`. Stages below cite `file:line` anchors.

**4.1 Logger initialisation** — `ExoplanetLightcurve.py:21`.
A file logger (`exo_console.log`, default level `logging.ERROR`) records the run.

**4.2 Master calibration frames** — `ExoplanetLightcurve.py:47` (dark),
`ExoplanetLightcurve.py:79` (bias).
Each calibration directory is read, the frames are stacked into a cube, and
combined pixel-wise. The **master dark** uses mean (`combo_master_dark == 1`) or
median (`== 2`) combination (`ExoplanetLightcurve.py:67`); the **master bias**
uses a mean (`ExoplanetLightcurve.py:99`). Both are saved as diagnostic PNGs.

**4.3 Per-star flux extraction — `fluxtarget()`** — `ExoplanetLightcurve.py:117`.
Invoked once per star (science + two calibrators). For every science frame it:

1. **Calibrates** the raw frame according to the selected reduction mode
   (`ExoplanetLightcurve.py:128`): dark-only, dark+bias, bias-only, or none.
2. **Crops** a square postage-stamp window centred on the star's pixel
   coordinates (`ExoplanetLightcurve.py:142`).
3. Optionally **subtracts a pedestal** (median/average/min) or applies
   `sigma_clip` (`ExoplanetLightcurve.py:184`).
4. **Detects the source** with `DAOStarFinder`, using a threshold set as a
   multiple of a background statistic from `sigma_clipped_stats`
   (`ExoplanetLightcurve.py:202`).
5. Performs **aperture photometry** with a `CircularAperture` and a concentric
   `CircularAnnulus` (`ExoplanetLightcurve.py:212`), then subtracts the local
   sky: `residual = aperture_sum − (annulus_mean × aperture_area)`
   (`ExoplanetLightcurve.py:223`). The residual aperture sum is the recorded flux.
6. Applies **manual drift correction**: at frame indices listed in
   `exo_input_values.i`, the crop window is translated by `shift_x`/`shift_y`
   (`ExoplanetLightcurve.py:244`). This compensates for the absence of an
   implemented auto-tracking routine (see author's comment at
   `exo_input_values.py:110`).

The routine returns the per-frame flux list and saves a flux-vs-frame PNG.

**4.4 Differential light curve** — `ExoplanetLightcurve.py:319` onward.
`fluxtarget` is called for the science target and both calibrators. The observation
time of each frame is read from the FITS `TIME-OBS` header keyword
(`ExoplanetLightcurve.py:350`), results are optionally written to CSV
(`ExoplanetLightcurve.py:357`), and the differential curves `F_sci/F_cal1` and
`F_sci/F_cal2` are formed (`ExoplanetLightcurve.py:370`).

**4.5 Transit depth from median baselines** — `ExoplanetLightcurve.py:386`.
The median of the first 20 points (pre-transit baseline), last 20 points
(post-transit baseline), and the central 20 points (in-transit) are computed; the
depth is the difference between the out-of-transit and in-transit medians.

**4.6 Planetary parameter derivation** — `ExoplanetLightcurve.py:431`.
From the transit depth the pipeline computes the depth in magnitudes, the planet
radius `R_p = sqrt(ΔF · (R_* · R_⊙)²)` (`ExoplanetLightcurve.py:440`) and its
error, the bulk **density** and its error (`ExoplanetLightcurve.py:447`), and the
orbital **inclination** from the transit-duration geometry
(`ExoplanetLightcurve.py:453`). All results are printed to stdout.

**4.7 Plotting** — `ExoplanetLightcurve.py:469` onward.
Differential light curves (`sci-cal1`, `sci-cal2`), optional comparison/delta
plots, and predicted transit start/end markers are rendered to PNG.

## 5. Configuration (`exo_input_values.py`)

All behaviour is controlled by editing `exo_input_values.py`, a plain Python
module imported as a namespace (`import exo_input_values`). It is *configuration
as code*, not a data file. Key groups:

- **Paths** — `lights_filepath`, `darks_filepath`, `bias_filepath`,
  `save_images_filepath`.
- **Star identity & geometry** — `sci_name`/`cal1_name`/`cal2_name`,
  `sci_coordinates`/`cal1_coordinates`/`cal2_coordinates` (`[x, y]` pixels),
  `pix_around_star` (half-width of the crop window, default 25).
- **Calibration combination** — `combo_master_dark`, `combo_master_bias`
  (`1` = average, `2` = median).
- **Reduction mode flags** — `no_red`, `dark_red`, `bias_red`, `dark_bias_red`,
  `bias_min`, `bias_sigma`, `sigma_red`, plus per-frame cuts `median_cut`,
  `average_cut`, `min_cut`. These are **mutually exclusive**: enable exactly one
  by setting it to `1`. The active default is `dark_red = 1` (dark subtraction only).
- **Detection & photometry** — `background_sigma = [3, 5]`, `fwhm = 3.`,
  `threshold = [20., 2]` (multiplier, and an index selecting mean/median/stddev),
  `aperture = 4.`, `annulus = [6., 8.]` (inner/outer radii), `methods = ['subpixel']`.
- **Manual tracking** — `i` (frame indices at which to shift), `shift_x`, `shift_y`.
- **Transit window** — `starttransit_pred`, `endtransit_pred`, `pred_start`,
  `pred_end`.
- **CSV I/O** — `write_csv`, `write_csv_name`, `option_compare`,
  `compare_csv_filename`.
- **System parameters** — `rstar`, `e_rstar`, `a` (semi-major axis, AU),
  `P` (period, days), `m_planet`, `e_m_planet`, `trandur` (transit duration, min).
- **Physical constants** (below the "DO NOT EDIT" marker) — `rjup`, `rastron`
  (solar radius, m), `au`, `den_jup`, `m_jup`.

**Switching targets.** The file ships an active WASP-52 b block plus commented
parameter blocks for HAT-P-19 b, TrES-5 b, and KELT-16 b. Change target by
commenting out the current block and uncommenting the desired one; there is no
target-selection argument.

## 6. Running the pipeline

```bash
# 1. Populate the input directories with FITS frames, e.g.
#    _WASP52b/WASP52b/*.fits   (science lights)
#    _WASP52b/Dark/*.fits      (darks)
#    _WASP52b/Bias/*.fits      (biases)
# 2. Adjust exo_input_values.py (paths, coordinates, reduction mode, system data)
# 3. Run:
python ExoplanetLightcurve.py
```

**Inputs.** FITS images read via `astropy.io.fits`; frames within a category must
share dimensions (they are stacked into cubes). Each science frame's header must
contain a `TIME-OBS` keyword. An optional comparison CSV may be supplied.

**Outputs.**
- A results CSV (`write_csv_name`, e.g. `WASP52b-std.csv`) with columns
  `NUMBER, TIME-OBS, FLUX-SCI, FLUX-CAL1, FLUX-CAL2`.
- PNG figures in `save_images_filepath`: master dark/bias, per-frame images, flux
  graphs, and differential light curves.
- A log file `exo_console.log`.
- Derived planetary parameters (radius, density, inclination, and errors) printed
  to stdout.

**Dependencies** (no manifest is committed; install manually):
`numpy`, `matplotlib`, `astropy`, `photutils`.

## 7. Conventions & caveats

- **Config-as-code, single active flag.** Reduction and output behaviour is
  selected by `0/1` flags in `exo_input_values.py`; enabling more than one
  mutually-exclusive flag yields undefined behaviour (first matching branch wins).
- **Runs on import.** The script has no `main()` guard and executes fully when run
  or imported. Avoid importing `ExoplanetLightcurve` for introspection.
- **Fixed three-star model.** Exactly one science target and two calibrators are
  hard-wired into the MAIN block.
- **Manual drift tracking.** Star centroids are not tracked automatically across
  the night; crop windows are shifted by hand via the `i`/`shift_x`/`shift_y`
  arrays, which must be tuned per data set.
- **Legacy API — will not run unmodified on a current stack.** The code targets
  the ~2018 photutils/astropy API: `iters=` in `sigma_clipped_stats`/`sigma_clip`
  (now `maxiters=`), `CircularAperture.area()`/`CircularAnnulus.area()` as methods
  (now the `.area` property), and a callable `DAOStarFinder`. Running on modern
  releases requires these substitutions. Pin an older environment or port the API.
- **No flat-fielding, no tests, no CI, no dependency manifest.** Treat results as
  first-order estimates and preserve the GPL v3 licence when redistributing.

## 8. Glossary

- **Bias frame** — zero-exposure readout capturing the electronic pedestal.
- **Dark frame** — exposure with a closed shutter capturing thermal (dark) current.
- **Master frame** — a combined (mean/median) calibration frame with reduced noise.
- **Aperture / annulus** — a circular region summing target flux, and a concentric
  ring sampling the local sky background subtracted from it.
- **Differential photometry** — target flux normalised by comparison-star flux to
  cancel atmospheric and instrumental common-mode variations.
- **Transit depth** — fractional flux decrement during transit, `≈ (R_p/R_*)²`.
- **Airmass** — the relative path length of starlight through the atmosphere; its
  variation is a dominant systematic that differential photometry mitigates.

# CLAUDE.md

Guidance for AI agents (and human contributors) working in this repository,
written in a scientific register. It describes the reduction methodology
implemented here, grounds it in the accompanying bachelor thesis, maps each
processing stage to the source code, and records the conventions and caveats
necessary to operate or extend the pipeline safely.

## Provenance

This software is the analysis pipeline of the bachelor thesis

> **"Photometrische Messungen von Exoplanetentransits — Einfluss der
> Reduktionsmethode auf die Ergebnisse"**
> (*Photometric measurements of exoplanet transits — Influence of the reduction
> method on the results*)
> Florian Ennemoser (mat. 01114897), Bachelor of Science, 2018.
> Supervisor: Dr. Thorsten Ratzka. Institut für Physik, Bereich Geophysik,
> Astrophysik und Meteorologie, Karl-Franzens-Universität Graz.

The thesis' central research question is **methodological**: *how does the choice
of CCD reduction method change the physical parameters one derives for a transiting
planet?* Accordingly, the code is deliberately built as a switchboard of
alternative reduction paths (see §7) rather than a single fixed recipe, and the
same program was reused across all five observed targets. The pipeline is based on
worked `astropy` examples and on IDL differential-photometry code provided by the
supervisor; its `aperture_photometry` results are reported in the thesis as
equivalent to the NASA IDL `APER` routine. Theory references cited by the thesis
include Seager (2010) *Exoplanets*, Haswell (2010) *Transiting Exoplanets*, and
Warner (2006) *A Practical Guide to Lightcurve Photometry and Analysis*.

## 1. Abstract

This repository implements a **ground-based differential aperture-photometry
pipeline** for the detection and characterisation of **transiting extrasolar
planets ("hot Jupiters")**. From a time series of CCD frames it extracts the
instrumental flux of a science target and two comparison ("calibrator") stars,
forms a differential light curve to suppress atmospheric and instrumental
systematics, measures the fractional flux decrement produced by a planetary
transit, and derives first-order physical parameters of the planet — radius, bulk
density, and orbital inclination — with propagated uncertainties. The distinctive
purpose is to run this reduction under **several alternative calibration schemes**
and compare the resulting planet parameters against catalogue values. Five transits
were observed at the Lustbühel Observatory (Graz); three (WASP-52 b, HAT-P-19 b,
TrES-5 b) yielded usable light curves. The implementation comprises two Python
modules and is driven entirely by a static configuration file; there is no
packaging, test suite, or dependency manifest.

## 2. Scientific background

**Transit photometry.** When a planet crosses the disc of its host star along the
line of sight it occults part of the stellar surface, producing a periodic dip in
the observed flux. To first order the fractional depth equals the squared
planet-to-star radius ratio,

```
ΔF / F ≈ (R_p / R_*)²                          (thesis Eq. 1)
```

so a measured depth combined with an estimate of `R_*` (from the host's spectral
class) yields the planetary radius. From the derived depth the code computes:

```
R_p = √((ΔF/F) · R_*²)                         (Eq. 18)   →  ExoplanetLightcurve.py:440
ρ_p = 3 M_p / (4 π R_p³)                        (density)  →  ExoplanetLightcurve.py:447
t_ges ≈ (P/π) · arcsin √(R_*²/a² − cos²i)       (Eq. 13)   →  inversion at :453
```

Orbital **inclination** `i` is obtained by inverting the transit-duration relation
(Eq. 13) once `t_ges` and `P` are known (`ExoplanetLightcurve.py:453`); a maximum
inclination is computed from the `R_p`/`R_*` uncertainties. Planet masses are taken
from the NASA Exoplanet Archive, and all uncertainties propagate by the Gaussian
error-propagation law.

**CCD calibration.** Raw frames carry an additive electronic pedestal (**bias**,
0 s exposure, closed shutter — sets the read-noise/zero level) and a
temperature-dependent thermal signal (**dark current**, matched exposure, closed
shutter). Subtracting master calibration frames raises the signal-to-noise ratio.
The reduction relations used are `Light_red = Light − Dark` (thesis Eq. 14) and,
when bias-normalised, `Light_red = Light − Bias − Dark_norm·t₂` (Eq. 16).
**Flat-fielding is not performed** by this pipeline (noted as future work).

**Differential photometry.** Absolute ground-based photometry is corrupted by
time-variable atmospheric transparency and airmass. Dividing the target flux by the
flux of nearby comparison stars cancels these common-mode signals: the normalised
ratio `F_target / F_cal` is ≈ 1 out of transit and drops below 1 during transit.
This pipeline forms `F_sci/F_cal1` and `F_sci/F_cal2` as two independent light
curves (`ExoplanetLightcurve.py:370`).

## 3. Observational context

All data were acquired at the **Lustbühel Observatory**, Graz (≈ 495 m altitude),
over September–December 2016.

| Item        | Value |
|-------------|-------|
| Telescope   | ASA (Austro Systeme Austria) **500 mm f/9 Cassegrain** |
| Detector    | **SBIG STF-8300** CCD, 3326 × 2504 px, Peltier-cooled to −10 °C |
| Binning     | 3 × 3 (→ 1124 × 850 px; 0.75″/px at ~2″ seeing) — except TrES-2 |
| Filter      | Sloan r′ |
| Exposure    | 20–90 s (typically 50–60 s); ~30 s readout/save |
| Coordinates | source pixel positions picked in **SAOImage DS9** |

**Targets** (all "hot Jupiters"). Each corresponds to a parameter block in
`exo_input_values.py` (active block, or a commented-out block to be swapped in):

| Target      | Status        | Notes |
|-------------|---------------|-------|
| WASP-52 b   | ✅ successful | K2V host, 140±20 pc, V≈12.0, transit depth ≈ 2.0%; **active config** |
| HAT-P-19 b  | ✅ successful | sloped baseline from differential extinction (see §10) |
| TrES-5 b    | ✅ successful | faint (V≈13.7) yet recovered |
| KELT-16 b   | ❌ failed     | first ~⅓ of frames out of focus → DAOStarFinder detected only background |
| TrES-2 b    | ❌ failed     | not 3×3 binned |

## 4. Repository structure

```
.
├── ExoplanetLightcurve.py   # Main procedural pipeline (single entry point)
├── exo_input_values.py      # Configuration module (all tunable parameters)
├── README.md                # One-line description
├── LICENSE                  # GNU GPL v3
└── CLAUDE.md                # This document
```

Runtime data directories are **not** version-controlled; they are created on demand
(`os.makedirs`) from the paths in `exo_input_values.py`. For the active WASP-52 b
configuration the expected layout is:

```
_WASP52b/
├── WASP52b/   # science ("light") FITS frames   (input)
├── Dark/      # dark calibration FITS frames     (input)
├── Bias/      # bias calibration FITS frames     (input)
└── images/    # generated PNG figures and graphs (output)
```

## 5. Methodology — pipeline stages

The program is a top-to-bottom procedural script executed as
`python ExoplanetLightcurve.py`. There is no `if __name__ == "__main__"` guard;
importing the module runs the full reduction. The only reusable routine is
`fluxtarget(...)`. Stages below cite `file:line` anchors.

**5.1 Logger initialisation** — `ExoplanetLightcurve.py:21`.
A file logger (`exo_console.log`, default level `logging.ERROR`) records the run.
On start-up, missing data directories are created and the run is expected to be
re-invoked after the operator copies frames in.

**5.2 Master calibration frames** — `ExoplanetLightcurve.py:47` (dark),
`ExoplanetLightcurve.py:79` (bias).
Each calibration directory is read, the frames are stacked into a cube, and combined
pixel-wise. The **master dark** uses mean (`combo_master_dark == 1`) or median
(`== 2`) combination (`ExoplanetLightcurve.py:67`); a median rejects hot pixels and
cosmic rays. The **master bias** uses a mean (`ExoplanetLightcurve.py:99`). Both are
saved as diagnostic PNGs.

**5.3 Per-star flux extraction — `fluxtarget()`** — `ExoplanetLightcurve.py:117`.
Invoked once per star (science + two calibrators). Source coordinates were determined
externally in SAOImage DS9 and passed via the config. For every science frame it:

1. **Calibrates** according to the selected reduction mode
   (`ExoplanetLightcurve.py:128`): dark-only, dark+bias, bias-only, or none.
2. **Crops** a square postage-stamp window centred on the star's pixel coordinates
   (`ExoplanetLightcurve.py:142`) so the whole array need not be processed.
3. Optionally **subtracts a pedestal** (median/average/min) or applies `sigma_clip`
   (`ExoplanetLightcurve.py:184`).
4. **Detects the source** with `DAOStarFinder`, using a threshold set as a multiple
   of a background statistic from `sigma_clipped_stats`
   (`ExoplanetLightcurve.py:202`).
5. Performs **aperture photometry** with a `CircularAperture` and a concentric
   `CircularAnnulus` (`ExoplanetLightcurve.py:212`), then subtracts the local sky
   (annulus median): `residual = aperture_sum − (annulus_mean × aperture_area)`
   (`ExoplanetLightcurve.py:223`). The residual aperture sum is the recorded flux.
   Default radii (aperture 4 px, annulus 6/8 px) gave good results in the thesis; an
   aperture of ~3·FWHM captures ≈ 90% of a star's light.
6. Applies **manual drift correction**: at frame indices listed in
   `exo_input_values.i`, the crop window is translated by `shift_x`/`shift_y`
   (`ExoplanetLightcurve.py:244`) to follow the star, compensating for the absence of
   an implemented tracking routine (author's comment at `exo_input_values.py:110`).

The routine returns the per-frame flux list and saves a flux-vs-frame PNG.

**5.4 Differential light curve** — `ExoplanetLightcurve.py:319` onward.
`fluxtarget` is called for the science target and both calibrators. The observation
time of each frame is read from the FITS `TIME-OBS` header keyword
(`ExoplanetLightcurve.py:350`), results are optionally written to CSV
(`ExoplanetLightcurve.py:357`), and the differential curves are formed
(`ExoplanetLightcurve.py:370`).

**5.5 Transit depth from median baselines** — `ExoplanetLightcurve.py:386`.
The median of the first 20 points (pre-transit baseline), last 20 points
(post-transit baseline), and the central 20 points (in-transit) are computed;
comparing medians (rather than extrema) is deliberate because the data are noisy.
The depth is the difference between the (averaged) out-of-transit baseline and the
in-transit median.

**5.6 Planetary parameter derivation** — `ExoplanetLightcurve.py:431`.
From the transit depth the pipeline computes the depth in magnitudes, the planet
radius `R_p` (`:440`) and its error, the bulk **density** and its error (`:447`),
and the orbital **inclination** and a maximum inclination (`:453`). All results are
printed to stdout.

**5.7 Plotting** — `ExoplanetLightcurve.py:469` onward.
Differential light curves (`sci-cal1`, `sci-cal2`), optional comparison/delta plots,
and predicted transit start/end markers are rendered to PNG.

## 6. Configuration (`exo_input_values.py`)

All behaviour is controlled by editing `exo_input_values.py`, a plain Python module
imported as a namespace (`import exo_input_values`). It is *configuration as code*,
not a data file. Key groups:

- **Paths** — `lights_filepath`, `darks_filepath`, `bias_filepath`,
  `save_images_filepath`.
- **Star identity & geometry** — `sci_name`/`cal1_name`/`cal2_name`,
  `sci_coordinates`/`cal1_coordinates`/`cal2_coordinates` (`[x, y]` pixels),
  `pix_around_star` (half-width of the crop window, default 25).
- **Calibration combination** — `combo_master_dark`, `combo_master_bias`
  (`1` = average, `2` = median).
- **Reduction mode flags** — `no_red`, `dark_red`, `bias_red`, `dark_bias_red`,
  `bias_min`, `bias_sigma`, `sigma_red`, plus per-frame cuts `median_cut`,
  `average_cut`, `min_cut`. These are **mutually exclusive**: enable exactly one by
  setting it to `1`. The active default is `dark_red = 1` (the "standard reduction").
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
parameter blocks for HAT-P-19 b, TrES-5 b, and KELT-16 b. Change target by commenting
out the current block and uncommenting the desired one; there is no target-selection
argument.

## 7. Reduction methods (the thesis' subject)

The comparison at the heart of the thesis exercises the following schemes; each is
selected purely through the mutually-exclusive flags in `exo_input_values.py`:

| Method                 | Selecting flag(s)                | Operation |
|------------------------|----------------------------------|-----------|
| No reduction           | `no_red = 1`                     | raw lights |
| **Standard**           | `dark_red = 1` (**default**)     | `Light − MasterDark` |
| Bias                   | `bias_red = 1`                   | `Light − MasterBias` |
| Bias + Median          | `bias_red`/`bias_min` + `median_cut = 1` | bias reduction then per-frame median subtraction |
| Reduced dark count     | `dark_red = 1`, few darks in the master | standard, fewer dark frames combined |

**Headline finding.** The **standard (dark) reduction is the most accurate**,
reproducing the NASA Exoplanet Archive / Exoplanet Transit Database values most
closely (for WASP-52 b: R_p ≈ 1.15 R_Jup, ρ ≈ 400 kg/m³, i ≈ 87.3°). The
**bias+median** scheme is *not* usable — it inflated the derived planet radius by
> 0.1 R_Jup. Reducing the number of dark frames biased the result only mildly. When
modifying reduction logic, treat "does the standard branch still reproduce catalogue
values?" as the acceptance criterion.

## 8. Running the pipeline

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
- Derived planetary parameters (radius, density, inclination, and errors) printed to
  stdout.

**Dependencies** (no manifest is committed; install manually):
`numpy`, `matplotlib`, `astropy`, `photutils`. The thesis environment was Python 3.6
under Anaconda/Spyder.

## 9. Conventions & caveats

- **Config-as-code, single active flag.** Reduction and output behaviour is selected
  by `0/1` flags in `exo_input_values.py`; enabling more than one mutually-exclusive
  flag yields undefined behaviour (first matching branch wins).
- **Runs on import.** The script has no `main()` guard and executes fully when run or
  imported. Avoid importing `ExoplanetLightcurve` for introspection.
- **Fixed three-star model.** Exactly one science target and two calibrators are
  hard-wired into the MAIN block.
- **Manual drift tracking is the acknowledged principal limitation.** No analog or
  digital star tracking is implemented; crop windows are shifted by hand via the
  `i`/`shift_x`/`shift_y` arrays, which must be re-tuned per data set. The failed
  KELT-16 b run is the cautionary example: with the first third of frames out of
  focus, `DAOStarFinder` locked onto background instead of the star, corrupting that
  segment of the light curve.
- **Airmass / baseline slope.** Because target and comparison stars have slightly
  different, wavelength-dependent atmospheric extinction, the out-of-transit
  differential flux drifts as the field rises or sets (clearly seen for HAT-P-19 b).
  The code therefore compares the in-transit median against the **mean of the pre-
  and post-transit baselines** rather than either endpoint alone. A proper linear
  baseline fit is listed as future work.
- **Legacy API — will not run unmodified on a current stack.** The code targets the
  ~2018 photutils/astropy API: `iters=` in `sigma_clipped_stats`/`sigma_clip` (now
  `maxiters=`), `CircularAperture.area()`/`CircularAnnulus.area()` as methods (now the
  `.area` property), and a callable `DAOStarFinder`. Running on modern releases
  requires these substitutions; pin an older environment or port the API.
- **No flat-fielding, no tests, no CI, no dependency manifest.** Treat results as
  first-order estimates and preserve the GPL v3 licence when redistributing.

## 10. Glossary

- **Bias frame** — 0 s readout capturing the electronic pedestal / read noise.
- **Dark frame** — matched-exposure, closed-shutter frame capturing thermal current.
- **Master frame** — a mean/median-combined calibration frame with reduced noise; a
  median additionally rejects hot pixels and cosmic rays.
- **Aperture / annulus** — a circular region summing target flux, and a concentric
  ring sampling the local sky background subtracted from it. Pixels between the two
  are ignored.
- **Differential photometry** — target flux normalised by comparison-star flux to
  cancel atmospheric and instrumental common-mode variation.
- **Transit depth** — fractional flux decrement during transit, `≈ (R_p/R_*)²`.
- **Airmass** — relative atmospheric path length of starlight; its (colour-dependent)
  variation as a field rises/sets is a dominant systematic that differential
  photometry mitigates.
- **Binning** — on-chip combination of pixels (here 3×3) that lowers spatial
  resolution but raises signal-to-noise and readout speed.

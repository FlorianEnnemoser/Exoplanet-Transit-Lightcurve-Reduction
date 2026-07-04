# Run Report — P2 data inventory & standard-reduction runs

Date: 2026-07-04 · Package `exotransit` 0.1.0 (astropy 6.1.7 / photutils 2.0.2 /
numpy 2.2.6, Python 3.13). Standard (dark-subtraction) reduction throughout.

This report is the human-readable stand-in for the committed regression fixture
(TODO line 78 / S-26): the WASP-52 b data is too large and proprietary to commit,
so `tests/test_acceptance.py` skips in CI and runs only locally. It documents the
local data and the CLI runs that demonstrate the acceptance invariant holds — the
gate that P2's legacy removal waits on.

## 1. Data inventory

Frames ship flat per target with suffix naming (`*.BIAS.FIT` / `*.DARK.FIT` /
rest = lights); `io_fits.discover` classifies by suffix.

| Target | Dir | Config | Total FIT | Bias | Dark | Lights | Frame dims | Status |
|--------|-----|--------|----------:|-----:|-----:|-------:|-----------|--------|
| WASP-52 b | `_WASP52b/` | `configs/wasp52b.toml` | 192 | 10 | 10 | 172 | uniform 850×1124 | **runs — invariant reproduced** |
| HAT-P-19 b | `_HATP19b/` | `configs/hatp19b.toml` | 197 | 10 | 10 | 177 | **mixed** | rejected at validation |
| TrES-5 b | `_TrES5b/` | `configs/tres5b.toml` | 130 | 10 | 10 | 110 | **mixed** | rejected at validation |
| KELT-16 b | `_KELT16b/` | — none — | 235 | — | — | — | not run (no config; failed in thesis) |
| TrES-2 b | `_TrES2b/` | — none — | 297 | — | — | — | not run (no config; not 3×3 binned) |

`_WASP52b/images/` also holds a `WASP52_lightcurve.csv` + `WASP52_result.json`
from the runs below (regenerated each run; not versioned).

### Aberrant frames (why HAT-P-19 b / TrES-5 b are rejected)

Both sets carry a few odd-sized frames at the **start** of the sequence
(telescope/focus warm-up before the CCD settled at 850×1124). `io_fits.discover`
(S-9 / R-18) refuses any set with inconsistent dimensions and lists every
offending file — correct strict behaviour, not a pipeline bug:

- **HAT-P-19 b** — 4 aberrant among 193 nominal:
  `00062452…` (122×123), `00062453/54/55…` (1276×1686).
- **TrES-5 b** — 5 aberrant among 125 nominal:
  `00061034/36/37/38…` (1276×1686), `00061035…` (70×26).

To reduce these two targets, cull those leading frames (a data-hygiene step,
out of scope here).

## 2. Commands used

```bash
uv run exotransit validate configs/wasp52b.toml
uv run exotransit reduce   configs/wasp52b.toml --dry-run
uv run exotransit reduce   configs/wasp52b.toml     # writes _WASP52b/images/{csv,json}
uv run exotransit reduce   configs/hatp19b.toml     # exit 3 (DataError)
uv run exotransit reduce   configs/tres5b.toml      # exit 3 (DataError)
uv run pytest tests/test_acceptance.py -v           # cross-check
```

(An `IERSWarning` about a failed `finals2000A.all` download appears on every run —
offline TLS/cert fallback to astropy's bundled IERS-B table; harmless.)

## 3. Results

### WASP-52 b — success

CLI stdout:

```
R_p   = 1.2198 +/- 0.0309 R_Jup
rho   = 336.1 +/- 25.8 kg/m^3
i     = 87.254 deg (max 87.568)
depth = 0.02409  
delta_mag = 0.03274
```

**Invariant check** vs the frozen acceptance reference in
`tests/test_acceptance.py` (R-21 / S-26):

| Quantity | This run | Frozen ref | Catalogue/thesis |
|----------|---------:|-----------:|-----------------:|
| R_p (R_Jup) | 1.2198 | 1.2198 | ≈1.15 |
| ρ (kg/m³) | 336.1 | 336.1 | ≈400 |
| i (deg) | 87.254 | 87.254 | ≈87.3 |

Exact match to the frozen reference; inclination reproduces catalogue to 0.05°.
`pytest tests/test_acceptance.py` → **1 passed** (runs, not skipped). The
standard branch still reproduces catalogue values — **the invariant holds.**

Run notes: 24 `NoDetectionsWarning` and one `Calibrator_2: 6/172 frames flagged`
(both within the detection-guard budget; well under the 50 % abort gate).

### HAT-P-19 b / TrES-5 b — rejected at validation

Both exit **3 (DataError)** listing the aberrant frames in §1. No reduction is
attempted — validation is fail-fast and precedes photometry.

### KELT-16 b / TrES-2 b — not run

No config ships for either (per the thesis both failed: KELT-16 b out-of-focus
early frames, TrES-2 b not 3×3 binned). Inventory only.

## 4. Provenance (P2 task 2 — R-24 / S-22) — done

`outputs.write_json` makes every run self-describing; no code was needed for this
task — it was already implemented alongside the P1 outputs work and is called
unconditionally by `pipeline.run()`. From `_WASP52b/images/WASP52_result.json`:

```json
"provenance": {
  "software": "exotransit",
  "version": "0.1.0",
  "config_path": "configs\\wasp52b.toml",
  "config_sha256": "21b8a265f226c85343d33937c7a53224275165011f469255abcee375c800ffe5",
  "input_files": { "lights": [172 files…], "darks": [10…], "bias": [10…] },
  "run_started_utc": "2026-07-04T10:47:51…",
  "run_finished_utc": "2026-07-04T10:48:02…"
}
```

Software version + config SHA-256 + the full input-file list satisfy R-24: a run
is reproducible from its sidecar alone.

## 5. Conclusion

- **P2 task 2 (provenance):** already implemented — done.
- **P2 task 1 (remove legacy):** deferred. The invariant is demonstrated locally
  (WASP-52 b), but legacy stays the reference until P1 correctness and a
  shareable regression gate land.
- **Data note for the maintainer:** only WASP-52 b is reducible as-is; culling the
  handful of leading warm-up frames would unblock HAT-P-19 b and TrES-5 b.

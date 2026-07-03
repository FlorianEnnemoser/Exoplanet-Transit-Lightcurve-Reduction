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
- The target runtime is a **current Python (≥ 3.11)**, whose standard library ships
  `tomllib` for reading TOML — no dependency needed to parse.
- The format should have **low ambiguity** (YAML's implicit typing and significant
  whitespace are a known source of surprises; JSON forbids comments).

### Decision

Use **TOML** as the configuration file format for the refactored pipeline: one config
file per target, and the format the web data-input application (Part II, W-11) emits
and re-loads.

### Consequences

- `+` Read with the standard-library `tomllib` on Python ≥ 3.11 — zero parse dependency.
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
- The runtime target is a currently-supported Python (≥ 3.10, NFR-3), which
  should be easy to provision.

### Decision

Use **`uv`** as the project and dependency manager, with a single
**`pyproject.toml`** manifest as the source of truth: PEP 621 `[project]`
metadata, constrained dependencies, `requires-python = ">=3.10"` (NFR-3), a
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

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

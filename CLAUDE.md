# CLAUDE.md

Seismic interpretation: horizons and fault polygons, SEG-Y volume reading by declared byte positions, pick-to-event checks against a baseline, and vertical seismic profile tables. Volume access in place is the job of `seismic_lib`; the reader here is a small one for checks.
Part of a family of subsurface modules (see `seismic_lib` and `stress_field`); cut out of the E&P data engine by `scripts/extract_library.py` of `ep_geoscience_analytics`.

@../../shared/idiom/CORE.md

Voice and structure of written work (README, docs) follow Idiom's CORE.md and house style: British English, impersonal, one creative headline then plain varied headings.

## How to work
- Lead with working code, then brief notes. Small ambiguities: state the assumption in one line and proceed.
- Flag geoscience assumptions explicitly: units, depth reference (RT / TVDSS / seabed), absolute vs effective stress, sign convention.
- Mention any new dependency before adding it (`uv add <pkg>`).
- Run tests, ruff and the publication gate after changes.

## Rules of this library
- Pure functions returning `(result, events)`; no file or network I/O except the configuration loader.
- Never import the host repository (`src`) or a knowledge store. Dependencies are the base and the libraries listed in `pyproject.toml`; `tests/test_architecture.py` enforces it.
- Settings are packaged defaults in `defaults/`; a default carries its level and a one-line basis; an author-and-year citation is encouraged.
- Figures live in `plots/` (matplotlib is an optional extra) and follow the domain's figure rules and the shared figure layer of `ep-physics-base`.
- Tests use small synthetic data. No real or licensed data, no paths of a machine.

## Commands (Windows, uv; run from repo root)
- Setup: `uv sync --all-extras`; tests: `uv run pytest`; lint: `uv run ruff check .`; gate: `uv run python scripts/publication_gate.py`

Read `docs/STATUS.md` first.

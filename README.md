# ep-seismic: Horizons, faults and volumes, and whether a pick sits on an event

Seismic interpretation: horizons and fault polygons, SEG-Y volume reading by declared byte positions, pick-to-event checks against a baseline, and vertical seismic profile tables. Volume access in place is the job of `seismic_lib`; the reader here is a small one for checks.

**Status: private, built public-ready.** The library follows the rules of a public one (no host-repository imports, no data, no private paths, an automated publication gate in `scripts/publication_gate.py`) and is published only when that gate passes. Licence: MIT with acknowledgement (`LICENSE`).

## Install

```
uv add git+https://github.com/AdmcCarthy/ep-seismic
```

Depends on: `ep-physics-base`. Optional extras:

none

## What is in it

| Module | What it does |
| :--- | :--- |
| `seismic_interp` | Horizons and faults of the Volve interpretation: catalogue statistics, the horizon at the position of a well pick (a mistie in depth or time), fault s |
| `seismic_volume` | Post-stack seismic volume (SEG-Y): geometry checked against the survey's own text header, traces at well positions, and the nearest amplitude extremum |
| `segy` | SEG-Y reader for post-stack volumes (no extra dependency: numpy only). |
| `reader_seismic_interp` | Readers of the Volve geophysical interpretation files: horizons (comma separated, inline / crossline / X / Y / Z), fault sticks (fixed width) and faul |
| `vsp` | Readers of VSP / check-shot time-depth tables (folder `08.VSP_VELOCITY` of the Volve log archive).  Pure: text in, frame out. |

Functions are pure: they take frames and parameters and return `(result, events)` where they change or flag data; nothing reads or writes files except the configuration loader.

## Settings and assumptions

Every setting has a packaged default in `src/ep_seismic/defaults/`. A project can override a file by providing `config/<same name>` in its working directory; the project file wins. A default is an assumption unless it says otherwise: its level (`placeholder`, `inferred`, `published`, `official`, `exact`) and a one-line basis travel with it, and a citation (author and year) is given where one exists. Libraries never import a knowledge store: a host that has one passes its values in as an overlay.

## Develop

```
uv sync --all-extras
uv run pytest            # 2 test files
uv run ruff check .
uv run python scripts/publication_gate.py
```

Tests use small synthetic data generated on the fly. No network, no real data.

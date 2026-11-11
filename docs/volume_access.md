# Volume access (formerly the repository seismic_lib)

Merged on 2026-11-11 into ep-seismic as the subpackage `ep_seismic.volume`. The environment variable names keep their `SEISMIC_LIB_` prefix and the index folder stays `~/.seismic_lib`, so existing sidecars and settings keep working.

Get hold of and view seismic data without ever handling the whole file.

Seismic files can run to terabytes. This library reads them **in place**, describes
their geometry with a tiny index, and extracts only what is needed: one inline, one
crossline, or one arbitrary line through the survey, as a small section that can be
plotted or kept.

## Principles
- The survey file is never copied, loaded whole or committed to Git.
- **Compact index.** A regular 3D grid is described by a few numbers (a 1M-trace
  survey is a 774-byte sidecar), with every trace verified against the model.
  Irregular data, 2D lines and gathers fall back to a per-trace index and say why.
- **Time vs depth, and what the samples are, must be stated.** SEG-Y headers do not
  say whether the vertical axis is time or depth, nor whether the values are
  amplitude, velocity or something else. This library never guesses: you state it
  (it is then remembered for that file), or the file's own name / a keyed header
  line such as `VERTICAL DOMAIN: TWT` does. Folder names and bare words in the text
  header are shown as hints only. No section can be made until both are settled.
- **The sample axis comes from the trace header, checked.** The first sample is the
  delay recording time; the time scalar in bytes 215-216 is applied only when it is a
  value SEG-Y allows (1, ±10 ... ±10000). segyio multiplies by whatever is there: the
  Volve ST10010 stack carries 3, which made its 4 ms axis start at 12 ms and put every
  time 8 ms late. Anything disregarded is listed in `src.axis_notes` and on the index.
- Extracts are small (MB, not GB). Keep only what is needed.
- Report what the headers say (which X/Y bytes, coordinate scalar, unit codes,
  suspicious values); never guess silently.
- SEG-Y (via `segyio`) first. OSDU selections, OpenVDS, ZGY etc. plug in behind the
  `SeismicSource` interface later.

## Usage
```python
from ep_seismic.volume import (
    make_domain,
    open_segy,
    plot_section,
    section_inline,
    section_path,
)

with open_segy(r"D:\seismic\survey.sgy") as src:
    idx = src.index()  # builds once (one pass), then reloads sidecar
    print(idx.meta.warnings)  # anything suspicious in the headers
    print(src.detect())  # what the file says about its domain, + hints

    # Say what the samples are, once; it is remembered for this file.
    src.set_domain(make_domain("time", "amplitude", reference="MSL"))

    line = section_inline(src, idx, 1200, window=(1000, 3000))  # window in ms
    line.save(r"D:\extracts\inline1200.npz")  # small, worth keeping

    path = section_path(src, idx, [(x0, y0), (x1, y1)])  # arbitrary line
    plot_section(path, attribution="Data: <owner>, <licence>")
```
Depth and velocity example: `make_domain("depth", "velocity", reference="TVDSS")`.
Units come from the SEG-Y measurement-system code when it is set, otherwise pass
`vertical_unit=` / `quantity_unit=`.

Arbitrary paths take `(x, y)` vertices, or `(inline, crossline)` with
`vertices_kind="ilxl"`. Each point on the path takes the **nearest trace**; there is
no interpolation, and the section's notes say so.

## When the headers are not standard
Some files (many VSP and processing-specific ones) keep inline, crossline or
coordinates in other trace-header bytes. The index warns (for example "all zero" or
"negative inline numbers"). To fix it:
```python
print(src.describe_headers())  # mapped fields at first/middle/last trace, plus the
# byte positions the text header CLAIMS (unverified)
print(src.header_hints())  # e.g. [("SRC POS X", 73, 76), ("CDP", 5, 8), ...]

from ep_seismic.volume import HeaderMap

src.set_header_map(HeaderMap(inline=9, crossline=21, cdp_x=181, cdp_y=185))
idx = src.index()  # rebuilt: an index never outlives the map it used
```
The corrected map is remembered for that file (`src.header_map_origin` says whether
the map is `default`, `user` or `remembered`). Text-header byte claims are shown but
never applied for you.

## Is it time or depth? A plausibility check
```python
c = src.check_domain(water_depth=80.0)  # a few hundred traces; never sets the domain
print(c.report())
print(c.against(src.require_domain()))  # consistent / INCONSISTENT / cannot tell
```
or `uv run python scripts/check_domain.py FILE.segy --water-depth 80 --state time`.

It looks at the sample interval, the seabed against a water depth (a Norwegian-shelf
profile of 60-500 m unless you give the real one) and the base of the survey, and says
**undetermined** when they cannot separate the two. Often they cannot: with an average
velocity of 2000 m/s one millisecond of TWT is one metre of depth, so a record ending at
3400 or a reservoir at 2700 reads as plausible either way. The water bottom separates
them only outside an overlap (for the 60-500 m profile, below 81 or above 500) unless you
give the water depth, when it is decisive (time puts the seabed at about 1.35 x the water
depth, depth at 1.0 x). The sample interval does the rest (1, 2, 4 ms against 5, 10, 12.5,
25 m, and a metre value in the microsecond field reads as a tiny interval). The profiles
and typical ranges are general knowledge about the shelf, not facts about your file.

Checked on the real Volve ST10010 stack (time, 4 ms): the check leans time on the interval
alone and cannot use the water bottom, because the cube is blank above about 200 ms (the
shallow data are muted, as in an angle stack). `tests/test_domain_check_volve.py` also runs it on a depth-stretched copy of those traces
(`v = 1700 + 0.5 z`): at 10 m samples it says undetermined, at 12.5 m it leans depth, and
the same numbers on a 4 ms axis lean time, i.e. only the axis changes the verdict. That
test reads the local Volve file (about 300 traces, outside the repo) and skips itself
without it. "PSDM" is a migration in the depth domain whose output is often converted
back to time, so the word never settles the axis. Treat "undetermined" and "leans" as
"you must say it yourself", which is what `set_domain` already requires. A seabed pick
that lies far below where the data start is rejected as a deeper reflector.

## Reading
`read_traces` keeps disk access sequential: it reads traces in file order, runs of
adjacent traces in one call, then returns them in the order you asked for (repeats
allowed). On a cold SSD this was about 1.6-1.9x faster for a crossline or a diagonal
path and no different for scattered or adjacent traces; expect more on slower
storage. On a warm cache it is a few milliseconds slower for scattered reads.

## Layout
- `source.py` `SeismicSource` interface. `segy.py` the SEG-Y implementation.
- `index.py` `LatticeIndex` (regular grid) and `TraceIndex` (per trace).
- `domain.py` vertical domain and quantity, detection, remembered choice.
  `domain_check.py` the time-vs-depth plausibility check (never decides).
- `section.py` sections along a line or path, save/load. `plot.py` plotting.

## Known limits
- The per-trace fallback index holds about 60 bytes per trace while building, so
  an irregular survey of ~10^8 traces needs several GB of RAM. Regular 3D surveys
  do not (memory is bounded by the chunk size).
- Only fixed-length-trace SEG-Y is supported (sample formats 1, 2, 3, 5, 8).
- Index building reads each trace header once, so a TB file is read at disk speed
  once, then never again for that file.
- Header layouts that do not follow SEG-Y rev 1 give warnings; see "When the headers
  are not standard". Only 4-byte inline/crossline/coordinate fields (2-byte scalar).
- Coordinates are used as stored. SEG-Y does not give the CRS; distances are in the
  file's coordinate units.

## Setup (Windows, uv)
```
uv sync
uv run pytest
uv run ruff check .
uv run ruff format .
```

## Tests
Tests use small synthetic SEG-Y files generated on the fly. No real or licensed
data is stored in this repo.

## Data attribution
Manual checks have used a small file from the Volve dataset, which is published by
Equinor ASA and the former Volve licence partners (ExxonMobil and Bayerngas) under the Volve Terms and
Conditions, which are based on CC BY 4.0 with two changes: the data may not be sold, and the licence covers all data in
the set (https://www.equinor.com/energy/volve-data-sharing). No Volve data is included in this repository.
Volve is provided "as is"; see its terms.

## Licence

See the LICENSE of this repository (MIT with acknowledgement).

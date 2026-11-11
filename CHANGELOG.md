# Changelog

## 1.0.0 (2026-11-11)

First public release.

## What is in it

- Horizon and fault readers, with the domain (time or depth) read from each file's own header.
- Comparisons of picks with the nearest amplitude event against a baseline.
- Vertical seismic profile tables.
- `ep_seismic.volume`: SEG-Y read in place, a compact index, sections along a line or path, a time-or-depth plausibility check (formerly the repository seismic_lib).

## Things to know

- Limits and settings that the code reads are packaged defaults and are assumptions unless their level says otherwise (see `KNOWLEDGE.md` and the defaults folder); a project file of the same name overrides them.
- Functions are pure: frames and parameters in, `(result, events)` out; nothing reads or writes files except the configuration loader.
- Licence: MIT with acknowledgement (see `LICENSE`).
- Data names from the Norwegian Offshore Directorate (Sodir, formerly NPD) appear as the sources the code was built against.

# Status of ep-seismic

**State.** Cut out of the E&P data engine on the date of the first commit: 5 modules, 2 test files. The engine switched over on 2026-11-11: it depends on this library (its old module paths are aliases of the modules here), and this repository is the source of truth.

## Decisions (Adam, 2026-11-11)
* Domain libraries, one per topic; this one is `ep-seismic`. Figures belong to their domain.
* Built public-ready; made public on 2026-11-11 after the gate passed (0 blockers). Licence MIT with acknowledgement.
* One-way design: the library never imports a knowledge store; assumptions travel as packaged defaults with a level, a one-line basis and an author-and-year citation where there is one.
* The hydrocarbon colours of the Sodir Geostandard are used as they are.

## Changes made when the code was cut out
* none.

## Tests that moved only in part (the rest use the engine and stay there)
* none.

## Couplings still to cut
* none: every `src` import was mapped.

## Next steps
* Add the knowledge identifiers used here to `KNOWLEDGE.md` and export the defaults from the knowledge store.
* Keep `scripts/publication_gate.py` at zero blockers (it runs in CI); read the warnings (Sodir, NPD, Equinor Design System names) when adding documents.

## Merge with seismic_lib (2026-11-11)
* Adam: merge the two. The repository seismic_lib (read-in-place SEG-Y access, compact index, sections, domain checks, plots; segyio) is now the subpackage ep_seismic.volume (version 0.2.0); its tests and the check_domain script came with it. seismic_lib is archived with a pointer.
* Still two readers: ep_seismic.segy (93 lines, declared byte positions, header facts for the volume checks) and ep_seismic.volume.segy (segyio, index). 	ests/test_readers_agree.py now shows that they agree on headers, traces (IBM and IEEE) and the line sets of a synthetic volume. They still differ on purpose: ep_seismic.segy takes DECLARED byte positions and reads the survey's own text-header facts (corner points, bin size, byte claims), which the volume reader does not; so the small reader is kept for the checks and only its overlapping part (traces, binary header) could be moved onto the volume reader.

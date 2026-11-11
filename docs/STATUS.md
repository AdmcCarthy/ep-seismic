# Status of ep-seismic

**State.** Cut out of the E&P data engine on the date of the first commit: 5 modules, 2 test files. The engine still holds the original code and remains the source of truth until it switches over to depend on this library.

## Decisions (Adam, 2026-11-11)
* Domain libraries, one per topic; this one is `ep-seismic`. Figures belong to their domain.
* Private until the publication gate passes; built public-ready. Licence MIT with acknowledgement.
* One-way design: the library never imports a knowledge store; assumptions travel as packaged defaults with a level, a one-line basis and an author-and-year citation where there is one.
* The hydrocarbon colours of the Sodir Geostandard are used as they are.

## Changes made when the code was cut out
* none.

## Tests that moved only in part (the rest use the engine and stay there)
* none.

## Couplings still to cut
* none: every `src` import was mapped.

## Next steps
* Switch the engine to depend on this library (replace its module by a thin import).
* Add the knowledge identifiers used here to `KNOWLEDGE.md` and export the defaults from the knowledge store.
* Run `scripts/publication_gate.py` and clear the warnings before publishing.

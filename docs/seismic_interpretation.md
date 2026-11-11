# Seismic interpretation: horizons and faults (Volve)

Code: `src/ingest/seismic_interp.py`, `src/processing/seismic_interp.py`; config `config/adapters/volve.yaml` section `seismic_interp`; switch `seismic_interpretation` (`scripts/volve_check.py --family seismic_interpretation`; the horizon nodes, 5.7 million, go to the WORK folder); marts `seis_horizon_catalogue`, `seis_horizon_well_check`, `seis_horizon_well_summary`, `seis_fault_summary`, `seis_fault_polygon_summary`, `seis_fault_polygon_vertex`, `seis_pick_fault_distance`; notebook 30 (`30_horizons_faults_wells`).

## 1. What is read
`Volve_Geophysical_Interpretations.zip`: **32 horizon files** (33 grids), time (TWT, ms) and depth (m below sea level, positive down), surveys **ST0202** (base survey) and **ST10010** (reprocessed; inline of ST10010 = 2 x inline of ST0202 - 1), nodes `inline, crossline, x, y, z` on a 12.5 m grid (5.7 million nodes); folders Official, Other, Multiples (first and second multiples of BCU and Shetland) and Timelapse (Hugin 4D horizons, field centre only). **Fault sticks** (66 named faults, 1,356 sticks, in time; README column layout) and **fault polygons** of the top (78) and base (72) of the Hugin Formation. The coordinates are ED50 / UTM 31N as the horizon and polygon files state it (EPSG:23031); the fault-stick README says only "UTM Zone 31N" (the same system is ASSUMED).

## 2. Checks (findings, never corrections)
- **Horizon against well pick** (347 pairs, 6 surfaces: Seabed, Utsira top, Shetland top, Ty top, Hod top, Hugin top and base): the horizon is read at the pick position (inverse-distance mean of the four nearest nodes, only within 30 m, an assumption) and compared with the pick (depth: pick depth below MSL minus horizon depth; time: pick TWT minus horizon TWT). **Depth:** median mistie near zero for every surface (seabed -0.6 m, Hugin top +0.1 m, Ty top -2.9 m, Hugin base -1.0 m); 57-100 % of the picks within 10 m; 23 of 132 comparisons exceed 30 m and come from 7 wells (15/9-F-11 B, F-14, 17, F-1 C and the three C-2 wells), up to 104 m (15/9-17, Hugin base). **Time:** medians -0.4 to -33 ms, 0-46 % within 5 ms; the base survey ST0202 fits the Ty top worse (-22 ms) than the reprocessed ST10010 (-9 ms).
- **Fault-marked picks against the fault polygons:** picks with the qualifier FP (faulted pick) lie nearer to a polygon than unmarked picks (Top: median 53 m, n=7, against 95 m, n=19, one-sided rank test p=0.05; Base: 9 m, n=6, against 80 m, n=16, p=0.01). Suggestive (few picks), not a test of the interpretation.

## 3. Data errors found (DQ-125)
1. The **Seabed depth file holds two horizons**: after its first 280,641 nodes (seabed, 83-103 m) a second header block starts and 235,613 nodes of a **time** horizon of the Hod Formation (2,410-2,686 ms) follow; read whole, it gives a seabed at 2.7 km. The reader splits the blocks by their headers; the second block duplicates the Hod time file.
2. The fault-polygon README gives 71 polygons for the base of the Hugin Formation; the file holds 72.
3. The fault-stick README states no datum for UTM zone 31N.
4. Fault polygons that are not valid rings (one per set) are repaired with a zero buffer and flagged `repaired`.

## 4. Limits and data to request
No velocity model or depth-conversion description, no fault throw or dip (sticks in time only), no horizon uncertainty, the seismic volumes (SEG-Y) are not read, the geomodel is not in these files. A mistie of tens of metres is common between a seismic depth conversion and a well pick; which is right cannot be said from the data. Wanted: the velocity model, the geomodel grids and properties, the fault-seal and compartment interpretation.

## Selectivity of the tolerance (2026-11-09)

Next to the share of picks within the tolerance, the horizon summary carries the share that cross-well pairings reach (each pick time paired with the horizon at every other well of the file) and their ratio (`null_share_within_tolerance`, `tolerance_lift`). For the time horizons the 5 ms tolerance is 4 to 12 times selective for the Hod and Hugin horizons and about as selective as chance for the seabed, Ty, Utsira and Shetland horizons; the misties are 10 to 30 ms wide, so the tolerance is a screening label and not a tie criterion.

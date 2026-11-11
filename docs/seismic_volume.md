# Seismic volume (Volve, ST10010)

Family `seismic_volumes` (`segy`, `seismic_volume`, adapter section `seismic_volume`, the analysis notebooks). The archive folder `Seismic` holds one SEG-Y file: the 2011 pre-stack depth migration of the ST10010 survey, final full-angle stack (3-41 degrees), shown in two-way time (`..._KIRCH_FULL_T`). The order summary lists 18 stack data sets (near, mid, far, full angle; time and depth; final and raw); only this one is delivered.

## What is read

| Item | How |
| :--- | :--- |
| Text header | 40 card images (EBCDIC): survey, processing sequence, processing grid corner points, bin size, azimuth, header word positions, coordinate multiplier |
| Binary header | sample interval, samples, format code 1 (IBM float), polarity word (0: not stated), revision, fixed length |
| Trace headers | inline (bytes 189), crossline (193), X (181), Y (185), coordinate scalar (71, -100), delay (109); positions are declared in the adapter and checked against the text header |
| Traces | read on request by trace number (picks, horizon nodes, sections); the whole volume is never loaded |

No SEG-Y library is used: the layout is read with numpy (the IBM conversion is tested on known values).

## Rules

* The grid is fitted to all trace positions by least squares and compared with the text header (bin size, azimuth, the four corner points); a difference above the tolerance is a warning, never a correction.
* Points are placed on the grid by the inverse of the fitted plane and rounded to the bin, within 9 m of the bin centre (assumption); no trace is made up for a missing cell (26 of 288,720 cells).
* **Polarity is not stated.** A peak or trough is the sign of the amplitude only. No impedance step, fluid or synthetic seismogram is derived.
* A pick (two-way time) or a horizon node is compared with the nearest amplitude extremum within 16 ms (assumption). The offset is compared with a **baseline**: the same traces at times shifted by -60, -40, +40 and +60 ms, because the extrema of a band-limited trace are only a few samples apart and a small offset alone proves nothing. A time with no extremum in the window counts as the window.
* Horizons are those of the same survey (adapter `horizons`, selected by file name); the Hugin horizons were interpreted on the near stack. Large horizons are thinned regularly to 40,000 nodes.

## Marts

`seis_volume_catalogue` (one row, geometry check), `seis_volume_pick_event` and `seis_volume_pick_summary`, `seis_volume_horizon_fit`. Node tables and sections stay in the work folder.

## Results (2026-11-09)

The grid fitted to the trace headers lies within 5 mm of the four header corner points (bin 12.5 m, azimuth 284.0 degrees, 288,694 traces, 45 km2). Auto-tracked horizons (Utsira top, Shetland, Ty, Hod, BCU) lie on an extremum 72 to 100 % of the time (baseline 41 to 75 %) with a single sign; the Hugin top and base only 54 and 49 % (baseline 45 and 43 %) with mixed signs. Of 16 pick surfaces with at least 5 picks in time, 4 (Seabed, Hordaland, Draupne, Heather sand) are clearly closer to an extremum than the baseline; the rest are not, and the Ty Formation picks are a few milliseconds from the nearest extremum on average (check-shot time against the volume's time conversion).

## Limits

Polarity, wavelet, processing report, velocity model and a well-tie report are not delivered (req.126-128); the other stacks and the 2002 base survey are not in the archive folder; the 4D and amplitude-versus-angle questions need them. Findings are in DQ-144.
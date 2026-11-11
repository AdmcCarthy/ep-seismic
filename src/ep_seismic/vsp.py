"""Readers of VSP / check-shot time-depth tables (folder `08.VSP_VELOCITY` of the Volve log archive).  Pure: text in, frame out.

Two layouts occur:
* `pairs`: a three-column table of measured depth, vertical depth below MSL and two-way time in ms (`TZV_TIME_CHECKSHOT_*.ASC`); the header says whether the depth is MD from the rotary table (`MD RKB`) or from sea level (`MDMSL`).
* `vsurv`: the operator's survey table with Fortran `D` exponents and the columns `MD/RKB`, `TVD/RKB`, `TVD/MSL` and `TC` (one-way vertical time corrected to MSL, s).  ASSUMPTION: TC is the one-way vertical time from MSL
  (the layout of the other tables defines it so, and the first row agrees with the water velocity: 85 m at 1,478 m/s = 0.0575 s); two-way time = 2000 x TC [ms].
Names and column labels that belong to one source are in `config/adapters/volve.yaml` (`vsp`), the layouts are detected here from the header text.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

_NUM = r"[-+]?\d+(?:\.\d+)?"


def _floats(line: str) -> list[float] | None:
    toks = line.replace("D", "E").replace("d", "e").split()
    try:
        return [float(t) for t in toks]
    except ValueError:
        return None


def parse_pairs(text: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The three-column table (MD, TVD below MSL, TWT ms).  Returns the frame and the header facts (`md_basis`: RKB or MSL as the header states it, `curve`: the curve name line)."""
    lines = text.replace("\r", "").split("\n")
    head = "\n".join(lines[:30])
    md_basis = "MSL" if re.search(r"MD\s*MSL", head, re.I) else ("RKB" if re.search(r"MD\s*RKB", head, re.I) else "not stated")
    curve = next((l.split(":", 1)[1].strip() for l in lines[:20] if l.lower().startswith("curve name") and ":" in l), "")
    rows = []
    for l in lines:
        v = _floats(l)
        if v is not None and len(v) == 3:
            rows.append(v)
    d = pd.DataFrame(rows, columns=["md_m", "tvdss_m", "twt_ms"])
    return d, {"layout": "pairs", "md_basis": md_basis, "curve": curve}


def parse_vsurv(text: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The operator's survey table (header row `RECNO MD/RKB TVD/RKB TVD/MSL ... TC ...`, Fortran D exponents): MD from the rotary table, TVD below MSL, two-way time = 2000 x TC."""
    lines = text.replace("\r", "").split("\n")
    hi = next(i for i, l in enumerate(lines) if "RECNO" in l and "MD/RKB" in l)
    cols = lines[hi].split()
    rows = [v for l in lines[hi + 1:] if (v := _floats(l)) is not None and len(v) == len(cols)]
    d = pd.DataFrame(rows, columns=cols)
    out = pd.DataFrame({"md_m": d["MD/RKB"], "tvdss_m": d["TVD/MSL"], "twt_ms": 2000.0 * d["TC"]})
    rkb = (d["TVD/RKB"] - d["TVD/MSL"]).median()
    return out, {"layout": "vsurv", "md_basis": "RKB", "curve": "operator VSP survey table", "rkb_from_table_m": float(rkb)}


def read_timedepth(name: str, text: str, pair_patterns: list[str], vsurv_patterns: list[str]) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """The table of one member when its file name matches a layout pattern of the adapter, else None.  Rows are sorted by depth; a table that is not monotonic in time is kept and the check says so."""
    if any(re.search(p, name, re.I) for p in pair_patterns):
        d, meta = parse_pairs(text)
    elif any(re.search(p, name, re.I) for p in vsurv_patterns):
        d, meta = parse_vsurv(text)
    else:
        return None
    return d.sort_values("md_m").reset_index(drop=True), meta

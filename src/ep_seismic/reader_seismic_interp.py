"""Readers of the Volve geophysical interpretation files: horizons (comma separated, inline / crossline / X / Y / Z), fault sticks (fixed width) and fault polygons.  Pure: text in, frame out.

Horizon file: lines starting with `#` are the export log; then a header of a few lines (survey, horizon name, data source, domain: TIME or DEPTH, sometimes a remark) and the data lines `inline,crossline,x,y,z`.  The survey name,
horizon name, source and domain are read from that header, and the cartographic system from the `#` log (the project states ED50 / UTM 31N, EPSG:23031, in the horizon and polygon files; the fault-stick README says only `UTM Zone 31N`,
so its datum is an ASSUMPTION taken from the other files).
Fault sticks: README layout (1-based columns): 3-12 X, 14-24 Y, 26-36 TWT [ms], 52 stick position (1 start, 2 middle, 3 end), 53-100 fault name.
Fault polygons: `Mapping Polygon <name> <id>`, then a line with the domain and the point count, then that many `x y z` lines (z = TWT [ms]).
"""
from __future__ import annotations

import io
import re
from typing import Any

import numpy as np
import pandas as pd

_DATA = re.compile(r"^\s*-?\d+(?:\.\d+)?,\s*-?\d")
_DATA5 = re.compile(r"^\s*-?\d+(?:\.\d+)?(?:,\s*-?\d+(?:\.\d+)?){4}\s*$")


def _header_meta(head_lines: list[str], crs: str) -> dict[str, Any]:
    head = [l.strip() for l in head_lines if l.strip() and not l.startswith("#") and l.strip() not in ("=", ",")]
    survey = next((h for h in head if re.fullmatch(r"ST\d+", h)), head[0] if head else "")
    rest = [h for h in head if h != survey]
    return {"survey": survey, "horizon": rest[0] if rest else "", "source": rest[1] if len(rest) > 1 else "", "domain": next((h for h in rest if h in ("TIME", "DEPTH")), ""), "crs_text": crs, "header": " | ".join(head)}


def parse_horizons(text: str) -> list[tuple[pd.DataFrame, dict[str, Any]]]:
    """The horizons of one file.  A file normally holds one horizon, but a header block can start again inside the file (the Seabed depth file carries a second horizon, a time horizon of the Hod Formation, after its first
    280,000 nodes): every block with data is returned with its own header facts (`survey`, `horizon`, `source`, `domain` as the header states it, `crs_text`, `header`), its `block` number and `skipped_lines` (non-data lines
    of the block that were not header lines)."""
    lines = text.replace("\r", "").split("\n")
    crs = next((l.split(":", 1)[1].strip() for l in lines[:200] if l.startswith("# Source cartographic system name")), "")
    blocks: list[tuple[list[str], list[str]]] = []
    head: list[str] = []
    data: list[str] = []
    for l in lines:
        if not l.strip() or l.startswith("#"):
            continue
        if _DATA5.match(l):
            data.append(l)
        else:
            if data:
                blocks.append((head, data))
                head, data = [], []
            head.append(l)
    if data:
        blocks.append((head, data))
    out = []
    for b, (h, dl) in enumerate(blocks, start=1):
        meta = _header_meta(h, crs)
        meta["block"] = b
        meta["skipped_lines"] = 0
        d = pd.read_csv(io.StringIO("\n".join(dl)), header=None, usecols=[0, 1, 2, 3, 4], names=["inline", "crossline", "x", "y", "z"], dtype=float)
        out.append((d.dropna().reset_index(drop=True), meta))
    return out


def parse_horizon(text: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The first horizon of a file (see `parse_horizons`)."""
    return parse_horizons(text)[0]

def parse_fault_sticks(text: str) -> pd.DataFrame:
    """Fault sticks by the README column layout: `x`, `y`, `twt_ms`, `stick_position` (1 start, 2 middle, 3 end), `fault`; `stick_no` counts the sticks of a fault in file order (a new stick starts at position 1 or when the fault name changes)."""
    rows = []
    for l in text.replace("\r", "").split("\n"):
        if len(l) < 60:
            continue
        try:
            rows.append((float(l[2:12]), float(l[13:24]), float(l[25:36]), int(l[51]), l[52:100].strip()))
        except ValueError:
            continue
    d = pd.DataFrame(rows, columns=["x", "y", "twt_ms", "stick_position", "fault"])
    new = d["stick_position"].eq(1) | d["fault"].ne(d["fault"].shift())
    d["stick_no"] = new.cumsum()
    return d


def parse_fault_polygons(text: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Polygons of a `New Mapping Polygon Set`: one row per vertex with `polygon_id`, the declared `n_points`, `x`, `y`, `twt_ms` and the header facts (`set_name`, `domain`, `declared_polygons`).  A polygon whose vertex count differs
    from the declared one is kept with the count that was read (`n_points_read`)."""
    lines = text.replace("\r", "").split("\n")
    meta: dict[str, Any] = {"set_name": "", "domain": "", "declared_polygons": np.nan}
    rows = []
    i = 0
    poly = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("UNKNOWN") and not meta["set_name"]:
            toks = l.split()
            meta["set_name"] = toks[1] if len(toks) > 1 else ""
            meta["domain"] = next((t for t in toks if t in ("TIME", "DEPTH")), "")
        if l.startswith("Mapping Polygon"):
            m = re.findall(r"\d+", l)
            pid = int(m[-1]) if m else poly
            decl = lines[i + 1].split()
            n = int(next((t for t in decl if t.isdigit() and int(t) > 0 and t != "0"), 0)) if len(decl) > 1 else 0
            pts = []
            for q in lines[i + 2:i + 2 + n]:
                v = q.split()
                if len(v) == 3:
                    try:
                        pts.append([float(x) for x in v])
                    except ValueError:
                        break
            for x, y, z in pts:
                rows.append((pid, n, len(pts), x, y, z))
            poly += 1
            i += 2 + max(n, 0)
            continue
        i += 1
    meta["declared_polygons"] = poly
    return pd.DataFrame(rows, columns=["polygon_id", "n_points", "n_points_read", "x", "y", "twt_ms"]), meta

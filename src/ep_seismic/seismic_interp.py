"""Horizons and faults of the Volve interpretation: catalogue statistics, the horizon at the position of a well pick (a mistie in depth or time), fault sticks and fault polygons, and the distance of well picks to the fault polygons.
Pure functions; only the pipeline and its stage modules (`src/pipeline.py`, `src/stages/`) reads and writes.

A horizon is an INTERPRETATION on a seismic survey (inline / crossline grid): its depth or time at a well is compared with the well pick of the same surface, and the difference is a finding, never a correction (the picks come from
the wells, the horizon from the seismic and a velocity model).  Horizon depth is taken as metres below mean sea level and positive downwards (the picks file has TVDSS negative downwards; the horizon at the well is compared with `-tvdss`).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSIONS = {"seis_horizon_catalogue": "1.0", "seis_horizon_well_check": "1.0", "seis_fault_summary": "1.0", "seis_fault_polygon_summary": "1.0", "seis_pick_fault_distance": "1.0"}


def grid_spacing(d: pd.DataFrame) -> float:
    """The node spacing in metres: the median distance between nodes that are neighbours in the crossline direction of one inline (a regular survey grid)."""
    s = d.sort_values(["inline", "crossline"])
    same = (s["inline"].diff() == 0) & (s["crossline"].diff() == 1)
    dx, dy = s["x"].diff()[same], s["y"].diff()[same]
    return float(np.median(np.hypot(dx, dy))) if same.any() else float("nan")


def horizon_stats(d: pd.DataFrame, meta: dict[str, Any], rel_path: str, kind: str) -> dict[str, Any]:
    """One catalogue row: where the file is, what its header says, the number of nodes, the extent in X and Y, the node spacing, the range of z and how many nodes lie more than 4 interquartile ranges from the median (kept, flagged)."""
    z = d["z"]
    q1, q3 = z.quantile(0.25), z.quantile(0.75)
    iqr = max(q3 - q1, 1e-6)
    return {"file": rel_path, "kind": kind, "survey": meta["survey"], "horizon": meta["horizon"], "source": meta["source"], "domain": meta["domain"], "crs_text": meta["crs_text"], "nodes": len(d), "skipped_lines": int(meta.get("skipped_lines", 0)),
            "x_min": d["x"].min(), "x_max": d["x"].max(), "y_min": d["y"].min(), "y_max": d["y"].max(), "node_spacing_m": grid_spacing(d), "z_min": z.min(), "z_median": z.median(), "z_max": z.max(),
            "z_extreme_nodes": int(((z < z.median() - 4 * iqr) | (z > z.median() + 4 * iqr)).sum()), "area_km2_nodes": len(d) * (grid_spacing(d) ** 2) / 1e6 if np.isfinite(grid_spacing(d)) else np.nan}


def horizon_at(d: pd.DataFrame, x: np.ndarray, y: np.ndarray, max_distance_m: float, k: int = 4) -> pd.DataFrame:
    """The horizon z at points: the inverse-distance weighted mean of the `k` nearest nodes, only when the nearest node is within `max_distance_m` (a horizon has holes at faults and outside the survey; no value is made up there).
    Returns `z` and `nearest_node_m`."""
    from scipy.spatial import cKDTree

    tree = cKDTree(d[["x", "y"]].to_numpy(float))
    dist, idx = tree.query(np.c_[x, y], k=k)
    zz = d["z"].to_numpy(float)[idx]
    w = 1.0 / np.maximum(dist, 1e-3)
    z = (zz * w).sum(axis=1) / w.sum(axis=1)
    near = dist[:, 0]
    return pd.DataFrame({"z": np.where(near <= max_distance_m, z, np.nan), "nearest_node_m": near})


def horizon_well_check(d: pd.DataFrame, meta_row: dict[str, Any], picks: pd.DataFrame, pick_name: str, max_distance_m: float) -> pd.DataFrame:
    """The picks of one surface (`pick_name`) against one horizon: the horizon z at the pick position and the difference.  In the depth domain `mistie_m` = pick depth below MSL (-tvdss) minus horizon depth (positive: the pick is deeper
    than the horizon); in the time domain `mistie_ms` = pick TWT minus horizon TWT.  Picks without position or value, and picks beyond `max_distance_m` of a node, are listed with the reason; the pick qualifier is kept."""
    p = picks[picks["surface_name"].eq(pick_name)].copy()
    out = pd.DataFrame({"well": p["well"].to_numpy(), "surface_name": pick_name, "qualifier": p["qualifier"].fillna("").to_numpy(), "easting": p["easting"].to_numpy(), "northing": p["northing"].to_numpy(),
                        "pick_depth_below_msl_m": (-p["tvdss"]).to_numpy(), "pick_twt_ms": p["twt"].to_numpy(), "file": meta_row["file"], "horizon": meta_row["horizon"], "domain": meta_row["domain"], "kind": meta_row["kind"]})
    has_xy = out["easting"].notna() & out["northing"].notna()
    h = pd.DataFrame({"z": np.nan, "nearest_node_m": np.nan}, index=out.index)
    if has_xy.any():
        h.loc[has_xy] = horizon_at(d, out.loc[has_xy, "easting"].to_numpy(float), out.loc[has_xy, "northing"].to_numpy(float), max_distance_m).to_numpy()
    out["horizon_z"], out["nearest_node_m"] = h["z"], h["nearest_node_m"]
    val = out["pick_depth_below_msl_m"] if meta_row["domain"] == "DEPTH" else out["pick_twt_ms"]
    out["mistie"] = val - out["horizon_z"]
    out["mistie_unit"] = "m" if meta_row["domain"] == "DEPTH" else "ms"
    out["not_checked_because"] = np.select([~has_xy, val.isna(), out["horizon_z"].isna()], ["pick has no position", "pick has no value in this domain", f"no horizon node within {max_distance_m:g} m (hole, fault or outside the survey)"], default="")
    return out


def null_share_within(c: pd.DataFrame, tol: float) -> float:
    """The share of cross-well pairings within the tolerance: each pick value (horizon z + mistie) is paired with the horizon z at EVERY OTHER well of the same file (all ordered pairs of different wells, exact, no random draw).  This is what the tolerance would accept
    by chance, so `share_within_tolerance` over this share says how selective the tolerance is (a share no larger than this is not agreement).  NaN with fewer than 2 wells."""
    z = c["horizon_z"].to_numpy(float)
    v = z + c["mistie"].to_numpy(float)
    w = c["well"].to_numpy()
    diff = np.abs(v[:, None] - z[None, :])
    other = w[:, None] != w[None, :]
    return float((diff[other] <= tol).mean()) if other.any() else float("nan")


def check_summary(chk: pd.DataFrame, tolerance: dict[str, float]) -> pd.DataFrame:
    """Per horizon file: picks checked, median and largest absolute mistie, and the share within the tolerance of its unit (assumed: `tolerance` by unit, m or ms), with the share that cross-well pairings would reach (`null_share_within_tolerance`) and their ratio (`tolerance_lift`)."""
    c = chk[chk["not_checked_because"].eq("")]
    if c.empty:
        return pd.DataFrame()
    g = c.groupby(["file", "horizon", "domain", "kind", "surface_name", "mistie_unit"]).agg(picks_checked=("mistie", "size"), mistie_median=("mistie", "median"), mistie_abs_max=("mistie", lambda s: float(s.abs().max())),
                                                                                        mistie_p10=("mistie", lambda s: float(s.quantile(0.1))), mistie_p90=("mistie", lambda s: float(s.quantile(0.9)))).reset_index()
    g["share_within_tolerance"] = [float((c[(c["file"] == f)]["mistie"].abs() <= tolerance[u]).mean()) for f, u in zip(g["file"], g["mistie_unit"])]
    g["null_share_within_tolerance"] = [null_share_within(c[c["file"] == f], tolerance[u]) for f, u in zip(g["file"], g["mistie_unit"])]
    g["tolerance_lift"] = g["share_within_tolerance"] / g["null_share_within_tolerance"].where(g["null_share_within_tolerance"] > 0)
    n_all = chk.groupby("file").size()
    g["picks_not_checked"] = g["file"].map(n_all) - g["picks_checked"]
    return g


def fault_summary(sticks: pd.DataFrame) -> pd.DataFrame:
    """One row per named fault: sticks, points, the extent in X and Y, the length of the box diagonal (a rough size, not a fault length) and the TWT range of the sticks."""
    g = sticks.groupby("fault").agg(sticks=("stick_no", "nunique"), points=("x", "size"), x_min=("x", "min"), x_max=("x", "max"), y_min=("y", "min"), y_max=("y", "max"), twt_min_ms=("twt_ms", "min"), twt_max_ms=("twt_ms", "max")).reset_index()
    g["extent_m"] = np.hypot(g["x_max"] - g["x_min"], g["y_max"] - g["y_min"])
    return g.sort_values("points", ascending=False).reset_index(drop=True)


def _polygon(g: pd.DataFrame):
    from shapely.geometry import Polygon

    return Polygon(list(zip(g["x"], g["y"])))


def polygon_table(poly: pd.DataFrame, set_name: str) -> pd.DataFrame:
    """One row per polygon: vertices, the area (shoelace; a polygon that is not valid is repaired with a zero buffer and flagged `repaired`), the centroid and the TWT range of its vertices."""
    rows = []
    for pid, g in poly.groupby("polygon_id"):
        pg = _polygon(g) if len(g) >= 3 else None
        valid = bool(pg is not None and pg.is_valid)
        if pg is not None and not valid:
            pg = pg.buffer(0)
        rows.append({"set": set_name, "polygon_id": int(pid), "vertices": len(g), "area_m2": float(pg.area) if pg is not None else np.nan, "centroid_x": float(pg.centroid.x) if pg is not None else np.nan, "centroid_y": float(pg.centroid.y) if pg is not None else np.nan,
                     "twt_min_ms": g["twt_ms"].min(), "twt_max_ms": g["twt_ms"].max(), "repaired": bool(pg is not None and not valid), "declared_vs_read": bool((g["n_points"] != g["n_points_read"]).any())})
    return pd.DataFrame(rows)


def pick_fault_distance(picks: pd.DataFrame, poly: pd.DataFrame, set_name: str, pick_name: str) -> pd.DataFrame:
    """For the picks of one surface: whether the pick position lies inside a fault polygon of the set and the distance to the nearest polygon edge [m] (0 inside), with the pick's own qualifier (`FP` faulted pick, `FO` faulted out, ...)
    so that the qualifier can be tested against the geometry.  Picks without position are listed without a distance."""
    from shapely.geometry import Point

    pg = [(int(pid), _polygon(g)) for pid, g in poly.groupby("polygon_id") if len(g) >= 3]
    pg = [(i, p if p.is_valid else p.buffer(0)) for i, p in pg]
    p = picks[picks["surface_name"].eq(pick_name)]
    rows = []
    for r in p.itertuples():
        if pd.isna(r.easting) or pd.isna(r.northing):
            rows.append({"well": r.well, "surface_name": pick_name, "qualifier": r.qualifier or "", "set": set_name, "inside_polygon": np.nan, "distance_to_polygon_m": np.nan, "polygon_id": np.nan})
            continue
        pt = Point(r.easting, r.northing)
        best = min(((p_.distance(pt), i) for i, p_ in pg), default=(np.nan, np.nan))
        inside = any(p_.contains(pt) for _, p_ in pg)
        rows.append({"well": r.well, "surface_name": pick_name, "qualifier": r.qualifier or "", "set": set_name, "inside_polygon": inside, "distance_to_polygon_m": 0.0 if inside else float(best[0]), "polygon_id": best[1]})
    return pd.DataFrame(rows)

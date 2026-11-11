"""Post-stack seismic volume (SEG-Y): geometry checked against the survey's own text header, traces at well positions, and the nearest amplitude extremum to a well pick or a horizon.
Pure functions; only the pipeline and its stage modules (`src/pipeline.py`, `src/stages/`) reads and writes.

A volume is a PROCESSED image (here the 2011 PSDM full stack converted to time): an amplitude is relative, its polarity is not stated in the open files (binary header polarity word = 0) and no wavelet, velocity model or
well-tie report is delivered.  A horizon or pick compared with an extremum of the trace is a finding, never a correction: the horizons are interpreted on this very volume, the picks come from the wells and a check shot.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ep_physics_base.events import dq_event

SCHEMA_VERSIONS = {"seis_volume_catalogue": "1.0", "seis_volume_pick_event": "1.0", "seis_volume_horizon_fit": "1.0", "seis_volume_well_trace": "1.0"}


def scaled_xy(th: pd.DataFrame) -> pd.DataFrame:
    """X and Y in metres: the stored integer times the coordinate scalar (negative = divide, positive = multiply, 0 = as stored), as SEG-Y defines it for bytes 71-72."""
    s = th["scalar"].to_numpy(float)
    f = np.where(s < 0, 1.0 / np.maximum(np.abs(s), 1.0), np.where(s > 0, s, 1.0))
    return th.assign(x=th["x_raw"] * f, y=th["y_raw"] * f)


def trace_grid(th: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    """A (inline, crossline) array of 0-based trace numbers (-1 = no trace) and the lowest inline and crossline.  A repeated (inline, crossline) keeps the first trace; repeats are counted in the geometry check."""
    il0, xl0 = int(th["inline"].min()), int(th["crossline"].min())
    g = np.full((int(th["inline"].max()) - il0 + 1, int(th["crossline"].max()) - xl0 + 1), -1, dtype=np.int64)
    first = th.drop_duplicates(["inline", "crossline"])
    g[first["inline"].to_numpy() - il0, first["crossline"].to_numpy() - xl0] = first["trace"].to_numpy()
    return g, il0, xl0


def affine_fit(th: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares map from (inline, crossline) to (X, Y): returns `A` (2x2) and `b` (2) with `[x, y] = A @ [il, xl] + b`."""
    m = np.c_[th["inline"].to_numpy(float), th["crossline"].to_numpy(float), np.ones(len(th))]
    coef, *_ = np.linalg.lstsq(m, th[["x", "y"]].to_numpy(float), rcond=None)
    return coef[:2].T, coef[2]


def volume_catalogue(file: str, bh: dict[str, Any], facts: dict[str, Any], th: pd.DataFrame, tol_m: float) -> tuple[pd.DataFrame, list[dict]]:
    """One row for the volume: what the headers say, the grid found in the trace headers, the fit of the grid to a plane (bins on a regular rotated grid), the bin size and azimuth found against the text header, and the distance of
    the four corner points the text header lists from the fitted grid.  A difference above `tol_m` (assumed) is a warning event, not a correction."""
    ev: list[dict] = []
    th = scaled_xy(th)
    g, il0, xl0 = trace_grid(th)
    n_il, n_xl = g.shape
    A, b = affine_fit(th)
    fit = th[["x", "y"]].to_numpy(float) - (th[["inline", "crossline"]].to_numpy(float) @ A.T + b)
    resid = np.hypot(fit[:, 0], fit[:, 1])
    bin_il, bin_xl = float(np.hypot(*A[:, 0])), float(np.hypot(*A[:, 1]))
    azimuth = float(np.degrees(np.arctan2(A[0, 1], A[1, 1])) % 360.0)           # bearing of the direction of increasing crossline (from north, clockwise)
    dup = int(th.duplicated(["inline", "crossline"]).sum())
    row: dict[str, Any] = {"file": file, "survey_text": " ".join(facts.get("title", [])), "data_type": facts.get("data_type"), "processing_date": facts.get("processing_date"), "format_code": bh["format_code"],
                           "samples": bh["samples"], "interval_ms": bh["interval_us"] / 1000.0, "first_sample_ms": float(th["delay_ms"].iloc[0]), "last_sample_ms": float(th["delay_ms"].iloc[0]) + (bh["samples"] - 1) * bh["interval_us"] / 1000.0,
                           "polarity_code": bh["polarity_code"], "traces": bh["traces"], "inline_min": int(th["inline"].min()), "inline_max": int(th["inline"].max()), "crossline_min": int(th["crossline"].min()), "crossline_max": int(th["crossline"].max()),
                           "grid_cells": n_il * n_xl, "missing_traces": int(n_il * n_xl - len(th) + dup), "duplicate_traces": dup, "x_min": th["x"].min(), "x_max": th["x"].max(), "y_min": th["y"].min(), "y_max": th["y"].max(),
                           "bin_inline_m": bin_il, "bin_crossline_m": bin_xl, "azimuth_deg": azimuth, "grid_fit_max_residual_m": float(resid.max()), "area_km2": float(len(th) * bin_il * bin_xl / 1e6),
                           "delay_varies": bool(th["delay_ms"].nunique() > 1), "samples_vary": bool((th["samples_in_header"] != bh["samples"]).any())}
    stated = facts.get("corners", {})
    worst = 0.0
    for k, c in stated.items():
        pred = A @ np.array([c["inline"], c["crossline"]], float) + b
        worst = max(worst, float(np.hypot(pred[0] - c["x"], pred[1] - c["y"])))
    row["text_corner_max_offset_m"] = worst if stated else np.nan
    row["text_bin_m"], row["text_azimuth_deg"] = facts.get("bin_inline_m"), facts.get("azimuth_deg")
    for name, want, got in (("bin_size", facts.get("bin_inline_m"), bin_il), ("azimuth", facts.get("azimuth_deg"), azimuth)):
        if want is not None and abs(want - got) > (0.5 if name == "bin_size" else 1.0):
            ev.append(dq_event("volve", "seismic_volume", f"{name}_differs_from_text_header", 1, "warning", [file], f"text header {want}, trace headers give {got:.2f}"))
    if stated and worst > tol_m:
        ev.append(dq_event("volve", "seismic_volume", "corner_points_differ_from_grid", len(stated), "warning", [file], f"largest offset {worst:.1f} m from the grid fitted to the trace headers (tolerance {tol_m:g} m)"))
    if row["missing_traces"] or dup:
        ev.append(dq_event("volve", "seismic_volume", "grid_not_complete", row["missing_traces"] + dup, "info", [file], f"{row['missing_traces']} cells without a trace, {dup} repeated (inline, crossline); the SEG-Y keeps the first of a repeat"))
    if row["delay_varies"] or row["samples_vary"]:
        ev.append(dq_event("volve", "seismic_volume", "trace_timing_varies", 1, "warning", [file], "recording delay or sample count differs between traces: time axes below assume the first trace"))
    return pd.DataFrame([row]), ev


def locate(x: np.ndarray, y: np.ndarray, th: pd.DataFrame, max_distance_m: float) -> pd.DataFrame:
    """The trace of the volume that holds each point: inline and crossline from the inverse of the fitted grid, rounded to the bin, the 0-based trace number (-1 when the bin is outside the volume or has no trace) and the distance
    from the point to the bin centre.  A point more than `max_distance_m` from its bin centre is not located (no trace is made up)."""
    th = scaled_xy(th)
    g, il0, xl0 = trace_grid(th)
    A, b = affine_fit(th)
    ilxl = (np.c_[np.asarray(x, float), np.asarray(y, float)] - b) @ np.linalg.inv(A).T
    il, xl = np.rint(ilxl[:, 0]).astype(int), np.rint(ilxl[:, 1]).astype(int)
    inside = (il >= il0) & (il < il0 + g.shape[0]) & (xl >= xl0) & (xl < xl0 + g.shape[1])
    tr = np.full(len(il), -1, dtype=np.int64)
    tr[inside] = g[il[inside] - il0, xl[inside] - xl0]
    centre = np.c_[il, xl].astype(float) @ A.T + b
    dist = np.hypot(centre[:, 0] - np.asarray(x, float), centre[:, 1] - np.asarray(y, float))
    tr[dist > max_distance_m] = -1
    return pd.DataFrame({"inline": il, "crossline": xl, "trace": tr, "distance_to_bin_m": dist})


def horizon_traces(h: pd.DataFrame, th: pd.DataFrame) -> np.ndarray:
    """The 0-based trace of each horizon node, by its inline and crossline (both are in the horizon file), -1 when the volume has no trace there."""
    g, il0, xl0 = trace_grid(th)
    il, xl = h["inline"].to_numpy().astype(int) - il0, h["crossline"].to_numpy().astype(int) - xl0
    ok = (il >= 0) & (il < g.shape[0]) & (xl >= 0) & (xl < g.shape[1])
    out = np.full(len(h), -1, dtype=np.int64)
    out[ok] = g[il[ok], xl[ok]]
    return out


def sample_at(traces: np.ndarray, t0_ms: float, dt_ms: float, t_ms: np.ndarray) -> np.ndarray:
    """Amplitude of each trace (row) at its own time `t_ms` by linear interpolation; NaN outside the trace."""
    k = (np.asarray(t_ms, float) - t0_ms) / dt_ms
    lo = np.floor(k).astype(int)
    ok = np.isfinite(k) & (lo >= 0) & (lo < traces.shape[1] - 1)
    lo = np.where(ok, lo, 0)
    r = np.arange(len(traces))
    w = k - lo
    return np.where(ok, traces[r, lo] * (1 - w) + traces[r, lo + 1] * w, np.nan)


def nearest_extremum(traces: np.ndarray, t0_ms: float, dt_ms: float, t_ms: np.ndarray, window_ms: float) -> pd.DataFrame:
    """For each trace (row) the local extremum (peak = positive sample above both neighbours, trough = negative sample below both) closest in time to `t_ms`, within +-`window_ms`.
    Returns `event_ms` (parabolic refinement), `offset_ms` (event minus t), `kind` (peak or trough, the sign of the amplitude only: the polarity convention is not stated), `amplitude` and `amplitude_at_t` (interpolated at t).
    No extremum within the window gives NaN and kind ''."""
    n, ns = traces.shape
    k0 = (np.asarray(t_ms, float) - t0_ms) / dt_ms
    half = int(np.ceil(window_ms / dt_ms))
    ev = np.full(n, np.nan)
    amp = np.full(n, np.nan)
    kind = np.array([""] * n, dtype=object)
    for i in range(n):
        if not np.isfinite(k0[i]):
            continue
        c = int(round(k0[i]))
        lo, hi = max(c - half, 1), min(c + half, ns - 2)
        if hi <= lo:
            continue
        s = traces[i]
        j = np.arange(lo, hi + 1)
        is_peak = (s[j] > 0) & (s[j] > s[j - 1]) & (s[j] >= s[j + 1])
        is_trough = (s[j] < 0) & (s[j] < s[j - 1]) & (s[j] <= s[j + 1])
        cand = j[is_peak | is_trough]
        cand = cand[np.abs(cand * dt_ms + t0_ms - t_ms[i]) <= window_ms]
        if not len(cand):
            continue
        m = cand[np.argmin(np.abs(cand - k0[i]))]
        a, b_, c_ = s[m - 1], s[m], s[m + 1]
        den = a - 2 * b_ + c_
        d = 0.5 * (a - c_) / den if den != 0 else 0.0
        ev[i], amp[i] = t0_ms + (m + d) * dt_ms, b_ - 0.25 * (a - c_) * d
        kind[i] = "peak" if b_ > 0 else "trough"
    return pd.DataFrame({"event_ms": ev, "offset_ms": ev - np.asarray(t_ms, float), "kind": kind, "amplitude": amp, "amplitude_at_t": sample_at(traces, t0_ms, dt_ms, np.asarray(t_ms, float))})


def chance_offset(traces: np.ndarray, t0_ms: float, dt_ms: float, t_ms: np.ndarray, window_ms: float, shifts_ms: list[float]) -> np.ndarray:
    """The distance to the nearest extremum that a time with no meaning would have: the same traces at the times `t_ms + shift` for each shift (a baseline).  The mean of the absolute offsets over the shifts, a time with no extremum in
    the window counting as `window_ms` (censored), so it is comparable with `censored_abs_offset`.  Extrema of a band-limited trace are only a few samples apart, so a small offset alone is not evidence of a tie."""
    t = np.asarray(t_ms, float)
    cols = [censored_abs_offset(nearest_extremum(traces, t0_ms, dt_ms, t + s, window_ms)["offset_ms"].to_numpy(), window_ms) for s in shifts_ms]       # offsets are measured from each shifted time
    return np.mean(np.vstack(cols), axis=0)


def censored_abs_offset(offset_ms: np.ndarray, window_ms: float) -> np.ndarray:
    """Absolute offset with 'no extremum in the window' counted as the window (a lower bound of the true distance)."""
    a = np.abs(np.asarray(offset_ms, float))
    return np.where(np.isnan(a), window_ms, a)


def pick_events(picks: pd.DataFrame, loc: pd.DataFrame, traces: np.ndarray, t0_ms: float, dt_ms: float, window_ms: float, shifts_ms: list[float] | None = None) -> pd.DataFrame:
    """Each pick with a TWT and a located trace: the nearest extremum within the window.  `picks` and `loc` are row-aligned; `traces` holds one row per located pick (in the order of the located picks).
    With `shifts_ms` the baseline of a time with no meaning is added (`chance_abs_offset_ms`, see `chance_offset`)."""
    ok = loc["trace"].to_numpy() >= 0
    base = pd.DataFrame({"well": picks["well"].to_numpy(), "surface_name": picks["surface_name"].to_numpy(), "qualifier": picks["qualifier"].fillna("").to_numpy(), "pick_twt_ms": picks["twt"].to_numpy(float),
                         "distance_to_bin_m": loc["distance_to_bin_m"].to_numpy(), "trace": loc["trace"].to_numpy()})
    res = nearest_extremum(traces, t0_ms, dt_ms, base.loc[ok, "pick_twt_ms"].to_numpy(), window_ms)
    out = pd.concat([base.reset_index(drop=True), pd.DataFrame(index=base.index, columns=res.columns)], axis=1)
    out.loc[ok, res.columns] = res.to_numpy()
    for c in ("event_ms", "offset_ms", "amplitude", "amplitude_at_t"):
        out[c] = pd.to_numeric(out[c])
    out["kind"] = out["kind"].fillna("")
    out["abs_offset_censored_ms"] = np.where(ok, censored_abs_offset(out["offset_ms"].to_numpy(), window_ms), np.nan)
    out["chance_abs_offset_ms"] = np.nan
    if shifts_ms and ok.any():
        out.loc[ok, "chance_abs_offset_ms"] = chance_offset(traces, t0_ms, dt_ms, base.loc[ok, "pick_twt_ms"].to_numpy(), window_ms, shifts_ms)
    out["not_checked_because"] = np.where(~ok, "no trace within the distance limit", np.where(out["event_ms"].isna(), f"no extremum within {window_ms:g} ms", ""))
    return out


def pick_event_summary(ev: pd.DataFrame, min_picks: int = 5) -> pd.DataFrame:
    """Per surface: picks checked, median and largest absolute offset, share of peaks among the events and the median amplitude relative to the RMS amplitude of the traces (a surface with fewer than `min_picks` is blank, not zero)."""
    c = ev[ev["not_checked_because"].eq("")]
    rows = []
    for s, g in ev.groupby("surface_name"):
        cg = c[c["surface_name"].eq(s)]
        located = g[g["trace"] >= 0]
        rows.append({"surface_name": s, "picks": len(g), "located": len(located), "checked": len(cg), "median_abs_offset_ms": cg["offset_ms"].abs().median() if len(cg) >= min_picks else np.nan, "max_abs_offset_ms": cg["offset_ms"].abs().max() if len(cg) >= min_picks else np.nan,
                     "median_censored_offset_ms": located["abs_offset_censored_ms"].median() if len(located) >= min_picks else np.nan, "median_chance_offset_ms": located["chance_abs_offset_ms"].median() if len(located) >= min_picks else np.nan,
                     "share_peak": float((cg["kind"] == "peak").mean()) if len(cg) >= min_picks else np.nan, "median_offset_ms": cg["offset_ms"].median() if len(cg) >= min_picks else np.nan})
    return pd.DataFrame(rows)


def horizon_fit(name: str, h: pd.DataFrame, traces: np.ndarray, t0_ms: float, dt_ms: float, window_ms: float, near_ms: float, shifts_ms: list[float] | None = None, chance_every: int = 4) -> tuple[dict[str, Any], pd.DataFrame]:
    """How a TWT horizon sits on the amplitudes of the volume it was interpreted on: per node the nearest extremum within the window; summary of the share of nodes with an extremum within `near_ms` (assumed 'on an event'), the median
    absolute offset, the share of peaks among events and the amplitude at the horizon.  Returns the summary row and the per-node table (inline, crossline, x, y, z, offset, kind, amplitude)."""
    res = nearest_extremum(traces, t0_ms, dt_ms, h["z"].to_numpy(float), window_ms)
    node = pd.concat([h[["inline", "crossline", "x", "y", "z"]].reset_index(drop=True), res], axis=1)
    has = node["event_ms"].notna()
    row = {"horizon": name, "nodes_sampled": len(node), "nodes_with_event": int(has.sum()), "share_on_event": float((node.loc[has, "offset_ms"].abs() <= near_ms).sum() / max(len(node), 1)),
           "median_abs_offset_ms": float(node["offset_ms"].abs().median()), "share_peak": float((node.loc[has, "kind"] == "peak").mean()) if has.any() else np.nan,
           "median_amplitude_at_horizon": float(node["amplitude_at_t"].median()), "window_ms": window_ms, "near_ms": near_ms, "chance_share_on_event": np.nan, "chance_median_offset_ms": np.nan}
    if shifts_ms:                                  # baseline: the same traces at times shifted by the given amounts (every `chance_every`-th node, for cost)
        sub = slice(None, None, chance_every)
        cols = [censored_abs_offset(nearest_extremum(traces[sub], t0_ms, dt_ms, h["z"].to_numpy(float)[sub] + s, window_ms)["offset_ms"].to_numpy(), window_ms) for s in shifts_ms]
        allc = np.concatenate(cols)
        row["chance_share_on_event"], row["chance_median_offset_ms"] = float((allc <= near_ms).mean()), float(np.median(allc))
    return row, node

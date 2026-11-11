"""Horizon, fault-stick and fault-polygon readers and the checks against well picks (synthetic data only)."""
import numpy as np
import pandas as pd

from ep_seismic import reader_seismic_interp as si
from ep_seismic import seismic_interp as sp

HOR = """#
# Source cartographic system name: ST_ED50_UTM31N_P23031_T1133
# comment line
=
ST10010
Seabed+VelBuild
PUB
DEPTH
,
100.0,200.0,1000.0,2000.0,90.5
100.0,201.0,1012.5,2000.0,91.5
101.0,200.0,1000.0,2012.5,92.5
ST10010
ST+ST10010+Hod_T
STAT
TIME
,
100.0,200.0,1000.0,2000.0,2500.0
100.0,201.0,1012.5,2000.0,2510.0
"""


def test_parse_horizons_splits_a_file_that_holds_two_horizons_and_reads_each_domain():
    blocks = si.parse_horizons(HOR)
    assert len(blocks) == 2
    (d1, m1), (d2, m2) = blocks
    assert len(d1) == 3 and m1["domain"] == "DEPTH" and m1["horizon"] == "Seabed+VelBuild" and m1["survey"] == "ST10010" and m1["crs_text"].startswith("ST_ED50_UTM31N")
    assert len(d2) == 2 and m2["domain"] == "TIME" and m2["horizon"] == "ST+ST10010+Hod_T" and m2["block"] == 2
    assert d1["z"].max() < 100 and d2["z"].min() > 2000                                           # the second block is not read as seabed depth


def test_fault_sticks_by_column_layout():
    line = lambda x, y, t, pos, name: f"  {x:10.3f} {y:11.3f}  {t:10.3f}  0 153   0  1 {pos}{name}".ljust(110)
    txt = "\n".join([line(435539.323, 6479337.382, 2602.036, 1, "FaultA"), line(435586.080, 6479383.795, 2711.715, 2, "FaultA"), line(435637.289, 6479434.629, 2814.798, 3, "FaultA"), line(435000.0, 6480000.0, 2700.0, 1, "FaultB")])
    d = si.parse_fault_sticks(txt)
    assert len(d) == 4 and d["fault"].tolist() == ["FaultA"] * 3 + ["FaultB"] and d["stick_no"].tolist() == [1, 1, 1, 2] and abs(d["twt_ms"].iloc[2] - 2814.798) < 1e-6


def test_fault_polygons_read_the_declared_vertices():
    txt = "New Mapping Polygon Set, version 1.1\nUNKNOWN    SetA   STAT  0  TIME\n 1 2 3 4\n\nMapping Polygon    UNKNOWN    2\n0         TIME           4     EXCLUSIVE   FAULT\n 0.0 0.0 100.0\n 10.0 0.0 100.0\n 10.0 10.0 100.0\n 0.0 10.0 100.0\nMapping Polygon    UNKNOWN    1\n0         TIME           3     EXCLUSIVE   FAULT\n 20.0 0.0 200.0\n 30.0 0.0 200.0\n 25.0 8.0 200.0\n"
    d, meta = si.parse_fault_polygons(txt)
    assert meta["declared_polygons"] == 2 and d["polygon_id"].nunique() == 2 and len(d) == 7
    t = sp.polygon_table(d, "S")
    assert abs(t[t["polygon_id"] == 2]["area_m2"].iloc[0] - 100.0) < 1e-9 and abs(t[t["polygon_id"] == 1]["area_m2"].iloc[0] - 40.0) < 1e-9


def _grid():
    x, y = np.meshgrid(np.arange(1000, 1100, 12.5), np.arange(2000, 2100, 12.5))
    return pd.DataFrame({"inline": np.repeat(np.arange(x.shape[0]), x.shape[1]).astype(float), "crossline": np.tile(np.arange(x.shape[1]), x.shape[0]).astype(float), "x": x.ravel(), "y": y.ravel(), "z": 3000.0 + 0.1 * (x.ravel() - 1000)})


def test_grid_spacing_and_horizon_at_with_a_distance_limit():
    g = _grid()
    assert abs(sp.grid_spacing(g) - 12.5) < 1e-9
    h = sp.horizon_at(g, np.array([1037.5, 5000.0]), np.array([2037.5, 5000.0]), 30.0)
    assert abs(h["z"].iloc[0] - (3000.0 + 3.75)) < 0.2 and np.isnan(h["z"].iloc[1])                    # far from every node: no value is made up


def test_horizon_well_check_depth_domain_sign_and_reasons():
    g = _grid()
    picks = pd.DataFrame({"well": ["A", "B", "C"], "surface_name": "Top", "qualifier": ["", "FP", ""], "easting": [1037.5, 1050.0, np.nan], "northing": [2037.5, 2050.0, np.nan], "tvdss": [-3010.0, -2990.0, -3000.0], "twt": [np.nan] * 3})
    meta = {"file": "f.dat", "horizon": "H", "domain": "DEPTH", "kind": "Official_Horizons"}
    c = sp.horizon_well_check(g, meta, picks, "Top", 30.0).set_index("well")
    assert c.loc["A", "mistie_unit"] == "m" and abs(c.loc["A", "mistie"] - (3010.0 - 3003.75)) < 0.2                   # pick deeper than the horizon: positive
    assert c.loc["C", "not_checked_because"] == "pick has no position" and c.loc["B", "qualifier"] == "FP"
    s = sp.check_summary(c.reset_index(), {"m": 10.0, "ms": 5.0})
    assert s["picks_checked"].iloc[0] == 2 and s["picks_not_checked"].iloc[0] == 1


def test_pick_fault_distance_inside_and_outside():
    poly = pd.DataFrame({"polygon_id": [1] * 4, "n_points": 4, "n_points_read": 4, "x": [0.0, 10.0, 10.0, 0.0], "y": [0.0, 0.0, 10.0, 10.0], "twt_ms": 100.0})
    picks = pd.DataFrame({"well": ["in", "out", "nopos"], "surface_name": "Top", "qualifier": ["FP", "", ""], "easting": [5.0, 30.0, np.nan], "northing": [5.0, 5.0, np.nan]})
    d = sp.pick_fault_distance(picks, poly, "Top", "Top").set_index("well")
    assert bool(d.loc["in", "inside_polygon"]) and d.loc["in", "distance_to_polygon_m"] == 0.0 and abs(d.loc["out", "distance_to_polygon_m"] - 20.0) < 1e-9 and np.isnan(d.loc["nopos", "distance_to_polygon_m"])


def test_tolerance_selectivity_against_cross_well_pairings():
    """Picks that equal the horizon at their own well but differ by 30 ms between wells: the tolerance accepts every own-well pairing and no cross-well pairing."""
    import pandas as pd

    chk = pd.DataFrame({"well": ["a", "b", "c"], "horizon_z": [2600.0, 2630.0, 2660.0], "mistie": [0.0, 0.0, 0.0]})
    assert sp.null_share_within(chk, 5.0) == 0.0
    assert abs(sp.null_share_within(chk, 40.0) - 4 / 6) < 1e-9 and sp.null_share_within(chk, 70.0) == 1.0
    assert pd.isna(sp.null_share_within(chk.iloc[:1], 5.0))
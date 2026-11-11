"""Seismic volume: SEG-Y reading (synthetic file), grid geometry against the text header, location of points, nearest extremum."""
import struct

import numpy as np
import pandas as pd

from ep_seismic import segy
from ep_seismic import seismic_volume as sv

WORDS = {"inline": 189, "crossline": 193, "x": 181, "y": 185}


def _write(path, n_il=4, n_xl=5, ns=60, fmt=5, skip=None):
    """A small stack: bins 12.5 m, inlines step in y, crosslines step in x; one Gaussian-modulated event at 100 ms (sample 4 ms from 0) whose sign flips with the inline."""
    text = ("C01 CLIENT : TEST   DATA TYPE: STACK-FULL (T)    DATE: 2011-10-26\n" + "C27 INLINE  BIN SIZE: 12.5M   CROSSLINE  BIN SIZE: 12.5M  AZIMUTH: 90 DEGREES\n"
            "C29 PG1   X: 1000.00   Y: 2000.00   IL:   10   XL: 100\n" "C35 3D INLINE  BYTES 189-192   |  3D CROSS LINE   BYTES 193-196\n" "C36 BIN CENTRE UTM-X  BYTES 181-184.  |  BIN CENTRE UTM-Y  BYTES 185-188\n" "C38 ALL COORDINATE VALUES ARE MULTIPLIED BY 100\n")
    lines = [l.ljust(80)[:80] for l in text.split("\n")[:-1]]
    head = "".join(lines).ljust(3200).encode("cp037")
    b = bytearray(400)
    struct.pack_into(">h", b, 16, 4000)
    struct.pack_into(">h", b, 20, ns)
    struct.pack_into(">h", b, 24, fmt)
    struct.pack_into(">h", b, 54, 1)
    struct.pack_into(">h", b, 300, 256)
    struct.pack_into(">h", b, 302, 1)
    t = np.arange(ns) * 4.0
    with open(path, "wb") as f:
        f.write(head)
        f.write(bytes(b))
        for i in range(n_il):
            for j in range(n_xl):
                if skip == (i, j):
                    continue
                th = bytearray(240)
                struct.pack_into(">i", th, 180, int(round((1000.0 + 12.5 * j) * 100)))
                struct.pack_into(">i", th, 184, int(round((2000.0 + 12.5 * i) * 100)))
                struct.pack_into(">i", th, 188, 10 + i)
                struct.pack_into(">i", th, 192, 100 + j)
                struct.pack_into(">h", th, 70, -100)
                struct.pack_into(">h", th, 108, 0)
                struct.pack_into(">h", th, 114, ns)
                s = ((-1.0) ** i * np.exp(-((t - 100.0) / 8.0) ** 2)).astype(">f4")
                f.write(bytes(th))
                f.write(s.tobytes())


def test_ibm_float_known_values():
    assert segy.ibm_to_float(np.array([0x42640000], dtype=np.uint32))[0] == 100.0
    assert segy.ibm_to_float(np.array([0xC276A000], dtype=np.uint32))[0] == -118.625
    assert segy.ibm_to_float(np.array([0], dtype=np.uint32))[0] == 0.0


def test_headers_geometry_and_text(tmp_path):
    p = str(tmp_path / "v.segy")
    _write(p, skip=(1, 2))
    lines = segy.read_text_header(p)
    facts = segy.text_header_facts(lines)
    assert facts["bin_inline_m"] == 12.5 and facts["azimuth_deg"] == 90.0 and facts["inline_byte"] == 189 and facts["coordinate_multiplier"] == 100
    bh = segy.read_binary_header(p)
    assert bh["samples"] == 60 and bh["interval_us"] == 4000 and bh["traces"] == 19 and bh["polarity_code"] == 0
    th = segy.read_trace_headers(p, bh, WORDS)
    cat, ev = sv.volume_catalogue(p, bh, facts, th, 5.0)
    r = cat.iloc[0]
    assert r["missing_traces"] == 1 and r["inline_min"] == 10 and r["crossline_max"] == 104
    assert abs(r["bin_inline_m"] - 12.5) < 1e-6 and abs(r["bin_crossline_m"] - 12.5) < 1e-6 and abs(r["azimuth_deg"] - 90.0) < 1e-6
    assert r["text_corner_max_offset_m"] < 1e-6
    assert any(e["rule"] == "grid_not_complete" for e in ev)


def test_locate_and_event(tmp_path):
    p = str(tmp_path / "v.segy")
    _write(p, skip=(1, 2))
    bh = segy.read_binary_header(p)
    th = segy.read_trace_headers(p, bh, WORDS)
    loc = sv.locate(np.array([1012.0, 1000.0, 5000.0, 1025.0]), np.array([2013.0, 2000.0, 2000.0, 2012.5]), th, 8.0)
    assert list(loc["trace"] >= 0) == [True, True, False, False]            # the 4th point falls in the missing bin; the 3rd is outside the volume
    assert loc["inline"].iloc[0] == 11 and loc["crossline"].iloc[0] == 101
    tr = segy.read_traces(p, bh, loc.loc[loc["trace"] >= 0, "trace"].to_numpy())
    # event at 100 ms: even inline is a peak, odd inline a trough; a pick 8 ms away is snapped back
    ev = sv.nearest_extremum(tr, 0.0, 4.0, np.array([108.0, 108.0]), 16.0)
    assert ev["kind"].tolist() == ["trough", "peak"] and np.allclose(ev["event_ms"], 100.0, atol=0.2) and np.allclose(ev["offset_ms"], -8.0, atol=0.2)
    far = sv.nearest_extremum(tr, 0.0, 4.0, np.array([200.0, 200.0]), 16.0)
    assert far["kind"].tolist() == ["", ""] and far["event_ms"].isna().all()
    assert abs(sv.sample_at(tr, 0.0, 4.0, np.array([100.0, 100.0]))[1] - 1.0) < 1e-5


def test_chance_baseline_is_larger_than_a_tie(tmp_path):
    """A pick on the event is closer to an extremum than the same trace read at shifted times (where the trace is empty, the window counts)."""
    p = str(tmp_path / "v.segy")
    _write(p)
    bh = segy.read_binary_header(p)
    tr = segy.read_traces(p, bh, np.arange(bh["traces"]))
    loc = pd.DataFrame({"trace": np.arange(len(tr)), "distance_to_bin_m": 0.0})
    picks = pd.DataFrame({"well": "w", "surface_name": "s", "qualifier": "", "twt": 102.0}, index=range(len(tr)))
    ev = sv.pick_events(picks, loc, tr, 0.0, 4.0, 16.0, [-40.0, 40.0])
    assert (ev["abs_offset_censored_ms"] < 3).all() and (ev["chance_abs_offset_ms"] == 16.0).all()
    s = sv.pick_event_summary(ev)
    assert s["median_censored_offset_ms"].iloc[0] < s["median_chance_offset_ms"].iloc[0]


def test_ibm_roundtrip_read(tmp_path):
    """A file written as IBM float is read back to the same values (format code 1)."""
    p = str(tmp_path / "v.segy")
    _write(p, n_il=1, n_xl=1, fmt=5)
    raw = bytearray(open(p, "rb").read())
    bh = segy.read_binary_header(p)
    vals = np.array([100.0, -118.625, 0.5, 0.0], dtype=np.float32)
    ibm = np.array([0x42640000, 0xC276A000, 0x40800000, 0], dtype=">u4")
    struct.pack_into(">h", raw, 3224, 1)
    off = 3600 + 240
    raw[off:off + 16] = ibm.tobytes()
    open(p, "wb").write(bytes(raw))
    bh = segy.read_binary_header(p)
    assert bh["format_code"] == 1
    assert np.allclose(segy.read_traces(p, bh, np.array([0]))[0, :4], vals)

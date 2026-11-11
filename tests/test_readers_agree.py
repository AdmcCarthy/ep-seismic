"""The two SEG-Y readers of this library (the small numpy reader of the checks, `ep_seismic.segy`, and the segyio reader of the volume code) agree on everything they share.

They exist for different jobs (declared byte positions and header facts against reading in place with an index); these tests are what must hold before one of them could be retired.
"""

import numpy as np
import pytest
import segyio
from segyio import TraceField

from ep_seismic import segy as small
from ep_seismic.volume import open_segy

ILINES, XLINES = [100, 101, 102], [200, 201, 202, 203]
SAMPLES = np.arange(0, 40, 4.0)


def make(path, fmt):
    spec = segyio.spec()
    spec.format, spec.ilines, spec.xlines, spec.samples = fmt, ILINES, XLINES, SAMPLES
    rng = np.random.default_rng(3)
    data = (rng.normal(size=(len(ILINES), len(XLINES), len(SAMPLES))) * 1000).astype(np.float32)
    with segyio.create(path, spec) as f:
        f.text[0] = segyio.tools.create_text_header({1: "AGREEMENT TEST   INLINE BIN SIZE: 25.0   FIRST SAMPLE: 0"})
        f.bin.update(hdt=4000)
        for i, il in enumerate(ILINES):
            for j, xl in enumerate(XLINES):
                n = i * len(XLINES) + j
                f.header[n] = {TraceField.INLINE_3D: il, TraceField.CROSSLINE_3D: xl, TraceField.CDP_X: 450000 + 25 * i, TraceField.CDP_Y: 6780000 + 12 * j, TraceField.SourceGroupScalar: -100}
                f.trace[n] = data[i, j]
    return data.reshape(-1, len(SAMPLES))


@pytest.mark.parametrize("fmt", [1, 5], ids=["IBM float", "IEEE float"])
def test_headers_and_traces_agree_with_segyio(tmp_path, fmt):
    path = str(tmp_path / f"agree{fmt}.sgy")
    data = make(path, fmt)
    bh = small.read_binary_header(path)
    assert (bh["samples"], bh["interval_us"], bh["format_code"], bh["traces"]) == (len(SAMPLES), 4000, fmt, len(ILINES) * len(XLINES))
    th = small.read_trace_headers(path, bh, {"inline": 189, "crossline": 193, "x": 181, "y": 185})
    with segyio.open(path, ignore_geometry=True) as f:
        assert th["inline"].tolist() == [h[TraceField.INLINE_3D] for h in f.header]
        assert th["crossline"].tolist() == [h[TraceField.CROSSLINE_3D] for h in f.header]
        assert th["x_raw"].tolist() == [h[TraceField.CDP_X] for h in f.header]
        assert th["scalar"].eq(-100).all()
        reference = f.trace.raw[:]
    idx = np.array([11, 0, 5, 5])
    mine = small.read_traces(path, bh, idx)
    np.testing.assert_allclose(mine, reference[idx], rtol=1e-6, atol=1e-4)           # same samples, IBM converted the same way
    np.testing.assert_allclose(reference, data, rtol=1e-6, atol=1e-2 if fmt == 1 else 0)   # and both equal what was written (IBM keeps about 24 bits)
    assert any("AGREEMENT TEST" in line for line in small.read_text_header(path))
    assert small.text_header_facts(small.read_text_header(path))["bin_inline_m"] == 25.0


def test_the_volume_index_sees_the_same_lines_as_the_small_reader(tmp_path):
    path = str(tmp_path / "grid.sgy")
    make(path, 5)
    bh = small.read_binary_header(path)
    th = small.read_trace_headers(path, bh, {"inline": 189, "crossline": 193, "x": 181, "y": 185})
    with open_segy(path, cache_dir=str(tmp_path / "cache")) as src:
        idx = src.index()
    assert idx.n_traces == len(th) == len(ILINES) * len(XLINES)
    assert sorted(set(th["inline"])) == ILINES and sorted(set(th["crossline"])) == XLINES
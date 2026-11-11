import os

import numpy as np
import pytest

from ep_seismic.volume import TraceIndex, index_path, load_index, open_segy
from ep_seismic.volume.segy import apply_scalar

from .conftest import ILINES, SAMPLES, XLINES, trace_values

N_TRACES = len(ILINES) * len(XLINES)


def test_open_reports_shape(make_segy) -> None:
    with open_segy(make_segy()) as src:
        assert src.n_traces == N_TRACES
        assert src.n_samples == len(SAMPLES)
        np.testing.assert_allclose(src.samples, SAMPLES)


def test_full_index_geometry_and_meta(make_segy) -> None:
    with open_segy(make_segy()) as src:
        idx = src.build_index()
    assert isinstance(idx, TraceIndex)
    assert idx.n_traces == N_TRACES
    assert idx.inline[:4].tolist() == [100] * 4
    assert idx.crossline[:4].tolist() == XLINES
    np.testing.assert_allclose(idx.x[:4], 1000.0)  # scalar -100 applied
    np.testing.assert_allclose(idx.y[:4], [5000.0, 5012.5, 5025.0, 5037.5])
    assert idx.meta.xy_fields == "cdp_x/cdp_y"
    assert idx.meta.sample_step == pytest.approx(4.0)
    assert idx.meta.measurement_system == 1
    assert idx.meta.coordinate_units == 1
    assert idx.meta.warnings == ()


@pytest.mark.parametrize(("scalar", "tol"), [(-100, 0.01), (10, 10), (0, 1), (1, 1)])
def test_scalar_signs_give_same_coordinates(make_segy, scalar, tol) -> None:
    # tol is the quantisation of the stored integers (positive scalars round).
    with open_segy(make_segy(scalar=scalar)) as src:
        idx = src.build_index()
    assert idx.x[0] == pytest.approx(1000.0, abs=tol)
    assert idx.y[1] - idx.y[0] == pytest.approx(12.5, abs=tol)


def test_scan_headers_chunk_boundaries(make_segy) -> None:
    with open_segy(make_segy()) as src:
        fields = {"inline": (189, 4), "scalar": (71, 2)}
        whole = src._scan_headers(fields, chunk=1_000)
        for chunk in (1, 5, N_TRACES - 1, N_TRACES):
            parts = src._scan_headers(fields, chunk=chunk)
            for name in fields:
                np.testing.assert_array_equal(parts[name], whole[name])
    assert whole["inline"].tolist() == [i for i in ILINES for _ in XLINES]
    assert set(whole["scalar"].tolist()) == {-100}


def test_little_endian_file(make_segy) -> None:
    with open_segy(make_segy(endian="little"), endian="little") as src:
        idx = src.build_index()
        data = src.read_traces(idx.select(inline=101))
    assert idx.crossline[:4].tolist() == XLINES
    assert idx.x[0] == pytest.approx(1000.0)
    np.testing.assert_array_equal(data[0], trace_values(1, 0))


def test_file_size_not_fitting_traces_is_refused(make_segy) -> None:
    path = make_segy()
    with open(path, "ab") as fh:
        fh.write(b"\0" * 7)  # trailing bytes: layout no longer provable
    with pytest.raises(ValueError, match="cannot open"):
        open_segy(path)


def test_apply_scalar_rules() -> None:
    vals = np.array([1000, 1000, 1000])
    out = apply_scalar(vals, np.array([-100, 10, 0]))
    np.testing.assert_allclose(out, [10.0, 10000.0, 1000.0])


def test_falls_back_to_source_xy_with_warning(make_segy) -> None:
    with open_segy(make_segy(use_cdp=False)) as src:
        idx = src.build_index()
    assert idx.meta.xy_fields == "source_x/source_y"
    assert any("source X/Y" in w for w in idx.meta.warnings)
    assert idx.x[0] == pytest.approx(1000.0)


def test_negative_line_numbers_warn(make_segy) -> None:
    # Real case: a Volve VSP file keeps other data in the inline/crossline bytes.
    with open_segy(make_segy(ilines=[-5, -4, -3])) as src:
        idx = src.build_index()
    assert any("negative inline" in w for w in idx.meta.warnings)


def test_zero_line_headers_warn(make_segy) -> None:
    with open_segy(make_segy(zero_lines=True)) as src:
        idx = src.build_index()
    assert any("all zero" in w and "inline" in w for w in idx.meta.warnings)


def test_select_inline_and_crossline(make_segy) -> None:
    with open_segy(make_segy()) as src:
        idx = src.build_index()
    il = idx.select(inline=101)
    assert idx.crossline[il].tolist() == XLINES
    assert set(idx.inline[il]) == {101}
    xl = idx.select(crossline=202)
    assert idx.inline[xl].tolist() == ILINES
    with pytest.raises(ValueError, match="no traces"):
        idx.select(inline=999)
    with pytest.raises(ValueError, match="exactly one"):
        idx.select()
    with pytest.raises(ValueError, match="exactly one"):
        idx.select(inline=100, crossline=200)


def test_full_index_locate_and_nearest(make_segy) -> None:
    with open_segy(make_segy(offsets=(0, 50))) as src:
        idx = src.build_index()
    tn = idx.locate(np.array([100, 102, 999]), np.array([200, 203, 200]))
    assert tn[2] == -1
    assert idx.inline[tn[0]] == 100 and idx.crossline[tn[0]] == 200
    assert idx.offset[tn[0]] == 0  # lowest trace number among offsets
    near, dist = idx.nearest(np.array([1000.2, 9999.0]), np.array([5000.1, 0.0]))
    assert near[0] == tn[0] and near[1] == -1
    assert dist[0] < 1
    assert idx.bin_size == pytest.approx(12.5)
    with pytest.raises(ValueError, match="whole numbers"):
        idx.locate(np.array([100.5]), np.array([200]))


def test_read_traces_values_and_order(make_segy) -> None:
    with open_segy(make_segy()) as src:
        idx = src.build_index()
        trace_nums = idx.select(inline=102)[::-1]  # reversed on purpose
        data = src.read_traces(trace_nums)
    assert data.dtype == np.float32
    assert data.shape == (len(XLINES), len(SAMPLES))
    for row, j in zip(data, reversed(range(len(XLINES))), strict=True):
        np.testing.assert_array_equal(row, trace_values(2, j))


def test_read_traces_sample_window(make_segy) -> None:
    with open_segy(make_segy()) as src:
        data = src.read_traces([5], sample_slice=slice(2, 6))
    np.testing.assert_array_equal(data[0], trace_values(1, 1)[2:6])


def test_read_traces_guards(make_segy) -> None:
    with open_segy(make_segy()) as src:
        with pytest.raises(ValueError, match="outside"):
            src.read_traces([N_TRACES])
        with pytest.raises(ValueError, match="outside"):
            src.read_traces([-1])
        with pytest.raises(ValueError, match="1-D"):
            src.read_traces([[0, 1]])
        with pytest.raises(ValueError, match="1-D"):
            src.read_traces([0.5])
        with pytest.raises(ValueError, match="cap"):
            src.read_traces(np.arange(N_TRACES), max_bytes=100)
        assert src.read_traces([]).shape == (0, len(SAMPLES))


def test_sidecar_roundtrip_and_reuse(make_segy, tmp_path) -> None:
    cache = tmp_path / "cache"
    path = make_segy()
    with open_segy(path) as src:
        built = src.index(cache_dir=cache, compact=False)
        sidecar = index_path(path, cache)
        assert sidecar.exists()
        mtime = sidecar.stat().st_mtime_ns
        again = src.index(cache_dir=cache, compact=False)
        assert sidecar.stat().st_mtime_ns == mtime  # reused, not rebuilt
    loaded = TraceIndex.load(sidecar)
    for name in ("inline", "crossline", "offset", "x", "y"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(built, name))
        np.testing.assert_array_equal(getattr(again, name), getattr(built, name))
    assert loaded.meta == built.meta


def test_stale_index_is_rebuilt(make_segy, tmp_path) -> None:
    cache = tmp_path / "cache"
    path = make_segy()
    with open_segy(path) as src:
        src.index(cache_dir=cache)
        sidecar = index_path(path, cache)
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        assert not load_index(sidecar).is_current_for(path)
        fresh = src.index(cache_dir=cache)
    assert fresh.is_current_for(path)


def test_corrupt_sidecar_is_rebuilt(make_segy, tmp_path) -> None:
    cache = tmp_path / "cache"
    path = make_segy()
    sidecar = index_path(path, cache)
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b"not an npz")
    with open_segy(path) as src:
        idx = src.index(cache_dir=cache)
    assert idx.n_traces == N_TRACES


def test_cache_dir_env_override(make_segy, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SEISMIC_LIB_CACHE", str(tmp_path / "envcache"))
    path = make_segy()
    with open_segy(path) as src:
        src.index()
    assert index_path(path).parent == tmp_path / "envcache"
    assert index_path(path).exists()


def test_index_path_distinguishes_same_name_in_different_folders(tmp_path) -> None:
    a, b = tmp_path / "a" / "s.sgy", tmp_path / "b" / "s.sgy"
    assert index_path(a, tmp_path) != index_path(b, tmp_path)


def test_sample_axis_starts_at_the_delay_and_ignores_an_invalid_time_scalar(
    make_segy,
) -> None:
    # SEG-Y rev 1 allows 1, +/-10, ... +/-10000 in bytes 215-216. The Volve ST10010
    # stack carries 3 there; segyio multiplies the 4 ms delay by it and puts the
    # axis 8 ms late. The axis must start at the delay and the index must say why.
    with open_segy(make_segy(delay=4, time_scalar=3)) as src:
        np.testing.assert_allclose(src.samples, SAMPLES + 4.0)
        assert any("time scalar 3" in n and "segyio" in n for n in src.axis_notes)
        idx = src.build_index()
    assert idx.meta.sample_start == pytest.approx(4.0)
    assert any("time scalar 3" in w for w in idx.meta.warnings)


@pytest.mark.parametrize(
    ("delay", "time_scalar", "start"),
    [(0, 0, 0.0), (4, 0, 4.0), (4, 1, 4.0), (4, 10, 40.0), (40, -10, 4.0), (0, 3, 0.0)],
)
def test_sample_axis_applies_only_valid_time_scalars(
    make_segy, delay, time_scalar, start
) -> None:
    with open_segy(make_segy(delay=delay, time_scalar=time_scalar)) as src:
        np.testing.assert_allclose(src.samples, SAMPLES + start)
        assert src.axis_notes == ()  # nothing to report when the headers are regular


def test_sample_axis_reports_a_text_header_that_disagrees(make_segy) -> None:
    from ep_seismic.volume.segy import sample_axis

    axis, notes = sample_axis(3, 4.0, 4, 3, "C11 FIRST SAMPLE: 4 MS   LAST SAMPLE")
    np.testing.assert_allclose(axis, [4.0, 8.0, 12.0])
    assert len(notes) == 1  # the text header agrees with the axis, so only the scalar
    _, notes = sample_axis(3, 4.0, 12, 0, "C11 FIRST SAMPLE: 4 MS")
    assert any("FIRST SAMPLE 4 MS" in n and "gives 12" in n for n in notes)
    _, notes = sample_axis(3, 4.0, 0, 0, "C11 FIRST SAMPLE= 0,  LAST SAMPLE= 4500")
    assert notes == ()

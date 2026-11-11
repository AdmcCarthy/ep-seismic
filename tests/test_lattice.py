import numpy as np
import pytest

from ep_seismic.volume import LatticeIndex, TraceIndex, index_path, load_index, open_segy

from .conftest import ILINES, XLINES, trace_values, true_xy

N = len(ILINES) * len(XLINES)

CASES = {
    "plain": {},
    "xline_major": {"order": "xline_major"},
    "rotated": {"theta": 0.6},
    "odd_steps": {"ilines": [10, 12, 14, 16], "xlines": [5, 10, 15]},
    "descending": {"ilines": [300, 299, 298], "xlines": [50, 48, 46, 44]},
}


@pytest.fixture(params=sorted(CASES))
def pair(request, make_segy, tmp_path):
    """(lattice, full) indexes of the same synthetic file, plus the file path."""
    kw = CASES[request.param]
    path = make_segy(f"{request.param}.sgy", **kw)
    with open_segy(path, cache_dir=tmp_path / "c") as src:
        lat, reason = src.build_lattice()
        full = src.build_index()
    assert lat is not None, reason
    return lat, full, path


def test_lattice_is_detected_and_described(pair) -> None:
    lat, full, _ = pair
    assert isinstance(lat, LatticeIndex)
    assert lat.n_traces == full.n_traces
    assert lat.max_xy_residual < 0.02  # coordinates stored to 0.01
    assert lat.n_il * lat.n_xl == lat.n_traces


def test_lattice_matches_full_index_for_every_trace(pair) -> None:
    lat, full, _ = pair
    tn = np.arange(lat.n_traces)
    il, xl = lat.ilxl(tn)
    np.testing.assert_array_equal(il, full.inline)
    np.testing.assert_array_equal(xl, full.crossline)
    x, y = lat.xy(tn)
    np.testing.assert_allclose(x, full.x, atol=0.02)
    np.testing.assert_allclose(y, full.y, atol=0.02)
    assert lat.inline_range == full.inline_range
    assert lat.crossline_range == full.crossline_range


def test_select_agrees_with_full_index(pair) -> None:
    lat, full, _ = pair
    for il in (full.inline.min(), full.inline.max()):
        np.testing.assert_array_equal(
            lat.select(inline=int(il)), full.select(inline=int(il))
        )
    for xl in (full.crossline.min(), full.crossline.max()):
        np.testing.assert_array_equal(
            lat.select(crossline=int(xl)), full.select(crossline=int(xl))
        )


def test_locate_agrees_and_rejects_off_grid(pair) -> None:
    lat, full, _ = pair
    np.testing.assert_array_equal(
        lat.locate(full.inline, full.crossline), np.arange(lat.n_traces)
    )
    il0, xl0 = int(full.inline[0]), int(full.crossline[0])
    off = lat.locate(
        np.array([il0, il0 + 0.5, il0 + 10_000]), np.array([xl0, xl0, xl0])
    )
    assert off[0] == 0 and off[1] == -1 and off[2] == -1


def test_nearest_agrees_with_full_index(pair) -> None:
    lat, full, _ = pair
    rng = np.random.default_rng(1)
    pick = rng.integers(0, lat.n_traces, 30)
    # Points near bin centres (within 20% of a bin) so there are no ties.
    jx = rng.uniform(-0.2, 0.2, 30) * lat.bin_size
    jy = rng.uniform(-0.2, 0.2, 30) * lat.bin_size
    qx, qy = full.x[pick] + jx, full.y[pick] + jy
    tn_l, d_l = lat.nearest(qx, qy)
    tn_f, _ = full.nearest(qx, qy)
    np.testing.assert_array_equal(tn_l, pick)
    np.testing.assert_array_equal(tn_f, pick)
    assert (d_l < 0.3 * lat.bin_size).all()


def test_nearest_outside_survey_is_minus_one(pair) -> None:
    lat, full, _ = pair
    far = lat.nearest(np.array([full.x.max() + 1000]), np.array([full.y.max() + 1000]))
    assert far[0][0] == -1
    near, _ = lat.nearest(
        np.array([full.x[3]]), np.array([full.y[3]]), max_distance=0.001
    )
    assert near[0] == 3


def test_xy_of_ilxl_fractional(pair) -> None:
    lat, full, _ = pair
    a = np.array([lat.il0, lat.il0])
    b = np.array([lat.xl0, lat.xl0 + lat.xl_step])  # neighbouring crosslines
    x, y = lat.xy_of_ilxl(a, b)
    tn = lat.locate(a, b)
    np.testing.assert_allclose(x, full.x[tn], atol=0.02)
    np.testing.assert_allclose(y, full.y[tn], atol=0.02)
    mid_x, mid_y = lat.xy_of_ilxl(a[:1], (b[:1] + b[1:]) / 2)
    np.testing.assert_allclose(mid_x, (x[0] + x[1]) / 2, atol=0.02)
    np.testing.assert_allclose(mid_y, (y[0] + y[1]) / 2, atol=0.02)


def test_index_returns_lattice_and_stores_a_tiny_sidecar(pair, tmp_path) -> None:
    _, full, path = pair
    cache = tmp_path / "cache"
    with open_segy(path) as src:
        idx = src.index(cache_dir=cache)
    assert isinstance(idx, LatticeIndex)
    sidecar = index_path(path, cache)
    assert sidecar.stat().st_size < 4000
    loaded = load_index(sidecar)
    assert loaded == idx
    assert LatticeIndex.load(sidecar) == idx
    with pytest.raises(ValueError, match="not a TraceIndex"):
        TraceIndex.load(sidecar)


def test_chunk_size_does_not_change_the_result(make_segy) -> None:
    with open_segy(make_segy()) as src:
        ref, _ = src.build_lattice()
        for chunk in (1, 5, N - 1):
            lat, _ = src.build_lattice(chunk=chunk)
            assert lat == ref


def test_compact_false_gives_full_index_and_replaces_cached_lattice(
    make_segy, tmp_path
) -> None:
    cache = tmp_path / "cache"
    with open_segy(make_segy()) as src:
        assert isinstance(src.index(cache_dir=cache), LatticeIndex)
        assert isinstance(src.index(cache_dir=cache, compact=False), TraceIndex)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"drop": ((1, 1),)}, "whole number|breaks"),
        ({"offsets": (0, 50)}, "single line"),
        ({"jitter": (1, 2, 5.0)}, "regular grid"),
        ({"ilines": [100], "xlines": [200, 201, 202, 203, 204]}, "single line|fewer"),
    ],
)
def test_irregular_files_fall_back_to_full_index_with_a_reason(
    make_segy, tmp_path, kwargs, reason
) -> None:
    path = make_segy(**kwargs)
    with open_segy(path, cache_dir=tmp_path / "c") as src:
        lat, why = src.build_lattice()
        assert lat is None
        assert any(word in why for word in reason.split("|")), why
        idx = src.index()
    assert isinstance(idx, TraceIndex)
    assert any("not a regular grid" in w for w in idx.meta.warnings)


def test_coordinate_tolerance_is_respected(make_segy) -> None:
    path = make_segy(jitter=(1, 2, 0.6))
    with open_segy(path) as src:
        strict, _ = src.build_lattice(xy_tolerance=0.1)
        loose, _ = src.build_lattice(xy_tolerance=1.0)
    assert strict is None
    assert loose is not None and 0.1 < loose.max_xy_residual <= 1.0


def test_source_xy_fallback_noted_on_lattice(make_segy) -> None:
    with open_segy(make_segy(use_cdp=False)) as src:
        lat, _ = src.build_lattice()
    assert lat.meta.xy_fields == "source_x/source_y"
    assert any("source X/Y" in w for w in lat.meta.warnings)


def test_lattice_reads_the_right_traces(make_segy) -> None:
    with open_segy(make_segy(order="xline_major")) as src:
        lat = src.index()
        data = src.read_traces(lat.select(crossline=202))
    for i, row in enumerate(data):
        np.testing.assert_array_equal(row, trace_values(i, 2))
    x, y = lat.xy(lat.select(crossline=202))
    exp = [true_xy(i, 2) for i in range(len(ILINES))]
    np.testing.assert_allclose(x, [e[0] for e in exp], atol=0.02)
    np.testing.assert_allclose(y, [e[1] for e in exp], atol=0.02)


def test_select_errors(make_segy) -> None:
    with open_segy(make_segy()) as src:
        lat = src.index()
    with pytest.raises(ValueError, match="no traces"):
        lat.select(inline=999)
    with pytest.raises(ValueError, match="no traces"):
        lat.select(crossline=999)
    with pytest.raises(ValueError, match="exactly one"):
        lat.select()

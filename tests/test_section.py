import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ep_seismic.volume import (  # noqa: E402
    DomainUnspecified,
    Section,
    make_domain,
    open_segy,
    plot_section,
    section_crossline,
    section_inline,
    section_path,
)

from .conftest import ILINES, SAMPLES, XLINES, trace_values, true_xy  # noqa: E402

DOM = make_domain("time", "amplitude")


@pytest.fixture
def src_idx(make_segy, tmp_path):
    with open_segy(make_segy(), domain=DOM, cache_dir=tmp_path / "c") as src:
        yield src, src.index()


@pytest.fixture(params=["lattice", "full"])
def kind(request):
    return request.param


@pytest.fixture
def both(make_segy, tmp_path, kind):
    """Same synthetic file through the lattice and the per-trace index."""
    with open_segy(make_segy(theta=0.4), domain=DOM, cache_dir=tmp_path / "c") as src:
        yield src, src.index(compact=(kind == "lattice"))


def test_inline_section_values_and_geometry(src_idx) -> None:
    src, idx = src_idx
    s = section_inline(src, idx, 101)
    assert s.data.shape == (len(XLINES), len(SAMPLES))
    for j, row in enumerate(s.data):
        np.testing.assert_array_equal(row, trace_values(1, j))
    assert set(s.inline.tolist()) == {101}
    assert s.crossline.tolist() == XLINES
    np.testing.assert_allclose(s.distance, [0, 12.5, 25, 37.5], atol=0.02)
    assert s.kind == "inline 101" and s.domain is DOM
    assert s.source_name == "synth.sgy"


def test_crossline_section_and_ranges(src_idx) -> None:
    src, idx = src_idx
    s = section_crossline(src, idx, 202, inline_range=(101, 102))
    assert s.inline.tolist() == [101, 102]
    for k, row in enumerate(s.data):
        np.testing.assert_array_equal(row, trace_values(k + 1, 2))
    np.testing.assert_allclose(s.distance, [0, 25], atol=0.02)
    t = section_inline(src, idx, 100, crossline_range=(202, 201))  # order ignored
    assert t.crossline.tolist() == [201, 202]
    with pytest.raises(ValueError, match="no traces within"):
        section_inline(src, idx, 100, crossline_range=(900, 950))
    with pytest.raises(ValueError, match="no traces"):
        section_inline(src, idx, 999)


def test_time_window(src_idx) -> None:
    src, idx = src_idx
    s = section_inline(src, idx, 100, window=(8, 20))
    np.testing.assert_allclose(s.samples, [8, 12, 16, 20])
    np.testing.assert_array_equal(s.data[0], trace_values(0, 0)[2:6])
    with pytest.raises(ValueError, match="outside the samples"):
        section_inline(src, idx, 100, window=(1000, 2000))


def test_section_needs_a_settled_domain(make_segy, tmp_path) -> None:
    with open_segy(make_segy(), cache_dir=tmp_path / "c") as src:
        idx = src.index()
        with pytest.raises(DomainUnspecified, match="set_domain"):
            section_inline(src, idx, 100)
        s = section_inline(src, idx, 100, domain=DOM)  # explicit override is fine
        assert s.domain is DOM
        src.set_domain(DOM, remember=False)
        assert section_inline(src, idx, 100).domain is DOM


def test_size_cap_is_enforced(src_idx) -> None:
    src, idx = src_idx
    with pytest.raises(ValueError, match="cap"):
        section_inline(src, idx, 100, max_bytes=10)


def test_path_along_an_inline_matches_the_inline(both) -> None:
    src, idx = both
    x0, y0 = true_xy(1, 0, 0.4)
    x1, y1 = true_xy(1, 3, 0.4)
    p = section_path(src, idx, [(x0, y0), (x1, y1)])
    i = section_inline(src, idx, 101)
    np.testing.assert_array_equal(p.trace_numbers, i.trace_numbers)
    np.testing.assert_array_equal(p.data, i.data)
    np.testing.assert_allclose(p.distance, i.distance, atol=0.05)
    assert any("nearest trace" in n for n in p.notes)


def test_path_in_ilxl_coordinates(both) -> None:
    src, idx = both
    p = section_path(src, idx, [(100, 200), (102, 203)], vertices_kind="ilxl")
    assert p.trace_numbers[0] == idx.locate(np.array(100), np.array(200))
    assert p.trace_numbers[-1] == idx.locate(np.array(102), np.array(203))
    assert (np.diff(p.distance) > 0).all()


def test_diagonal_path_takes_the_nearest_trace_each_point(both) -> None:
    src, idx = both
    xa, ya = true_xy(0, 0, 0.4)
    xb, yb = true_xy(2, 3, 0.4)
    p = section_path(src, idx, [(xa, ya), (xb, yb)], spacing=5.0)
    px = np.interp(p.distance, [0, np.hypot(xb - xa, yb - ya)], [xa, xb])
    py = np.interp(p.distance, [0, np.hypot(xb - xa, yb - ya)], [ya, yb])
    # Each chosen trace is the closest of ALL traces to its path point.
    allx, ally = idx.xy(np.arange(idx.n_traces))
    for k in range(p.n_traces):
        d = np.hypot(allx - px[k], ally - py[k])
        assert p.trace_numbers[k] == np.argmin(d)
    assert len(set(p.trace_numbers.tolist())) == p.n_traces  # repeats dropped


def test_path_partly_outside_is_trimmed_and_noted(both) -> None:
    src, idx = both
    xa, ya = true_xy(1, -4, 0.4)  # well before the first crossline
    xb, yb = true_xy(1, 3, 0.4)
    p = section_path(src, idx, [(xa, ya), (xb, yb)])
    assert any("outside the survey" in n for n in p.notes)
    assert p.crossline.tolist() == XLINES[: p.n_traces]  # starts inside, not at xl 196
    assert p.distance[0] == 0


def test_path_errors(both) -> None:
    src, idx = both
    with pytest.raises(ValueError, match="at least two"):
        section_path(src, idx, [(0.0, 0.0)])
    with pytest.raises(ValueError, match="zero length"):
        section_path(src, idx, [(1000.0, 5000.0), (1000.0, 5000.0)])
    with pytest.raises(ValueError, match="entirely outside"):
        section_path(src, idx, [(9e6, 9e6), (9e6 + 100, 9e6)])
    with pytest.raises(ValueError, match="vertices_kind"):
        section_path(src, idx, [(0, 0), (1, 1)], vertices_kind="lonlat")
    with pytest.raises(ValueError, match="spacing"):
        section_path(src, idx, [(1000.0, 5000.0), (1050.0, 5000.0)], spacing=-1)


def test_save_and_load_roundtrip(src_idx, tmp_path) -> None:
    src, idx = src_idx
    s = section_inline(src, idx, 101)
    out = s.save(tmp_path / "out" / "line101")
    assert out.suffix == ".npz" and out.exists()
    back = Section.load(out)
    np.testing.assert_array_equal(back.data, s.data)
    np.testing.assert_array_equal(back.samples, s.samples)
    assert back.domain == s.domain and back.kind == s.kind and back.notes == s.notes
    assert out.stat().st_size < 5000


def test_section_validates_shapes(src_idx) -> None:
    src, idx = src_idx
    s = section_inline(src, idx, 100)
    with pytest.raises(ValueError, match="one value per trace"):
        Section(**{**s.__dict__, "x": s.x[:-1]})
    with pytest.raises(ValueError, match="samples"):
        Section(**{**s.__dict__, "samples": s.samples[:-1]})


def test_plot_labels_scales_and_caption(src_idx) -> None:
    src, idx = src_idx
    s = section_inline(src, idx, 100)
    ax = plot_section(s, attribution="Data: Example Owner (CC BY 4.0)")
    assert ax.get_ylabel() == "TWT [ms]"
    assert "Distance" in ax.get_xlabel()
    img = ax.images[0]
    lo, hi = img.get_clim()
    assert lo == -hi and hi > 0 and img.get_cmap().name == "gray"
    assert img.get_array().shape == (len(SAMPLES), len(XLINES))
    texts = " ".join(t.get_text() for t in ax.figure.texts)
    assert "Example Owner" in texts and "inline 100" in texts and "user" in texts
    cbar_label = ax.figure.axes[-1].get_ylabel()
    assert cbar_label == "Amplitude [unitless]"
    plt.close("all")


def test_plot_velocity_and_depth_labels(make_segy, tmp_path) -> None:
    dom = make_domain("depth", "velocity", measurement_system=1, reference="TVDSS")
    with open_segy(make_segy(), domain=dom, cache_dir=tmp_path / "c") as src:
        s = section_inline(src, src.index(), 100)
    ax = plot_section(s, xaxis="crossline", caption=False)
    assert ax.get_ylabel() == "Depth TVDSS [m]"
    assert ax.figure.axes[-1].get_ylabel() == "Velocity [m/s]"
    assert ax.images[0].get_cmap().name == "viridis"
    assert ax.get_xlabel() == "Crossline"
    assert not ax.figure.texts
    plt.close("all")


def test_plot_guards(src_idx) -> None:
    src, idx = src_idx
    s = section_inline(src, idx, 100)
    with pytest.raises(ValueError, match="xaxis"):
        plot_section(s, xaxis="depth")
    with pytest.raises(ValueError, match="does not vary"):
        plot_section(s, xaxis="inline")
    with pytest.raises(ValueError, match="clip"):
        plot_section(s, clip=10)
    one = Section(
        **{
            **s.__dict__,
            **{
                k: getattr(s, k)[:1]
                for k in ("trace_numbers", "x", "y", "inline", "crossline", "distance")
            },
            "data": s.data[:1],
        }
    )
    with pytest.raises(ValueError, match="two traces"):
        plot_section(one)
    plt.close("all")


def test_plot_into_given_axes(src_idx) -> None:
    src, idx = src_idx
    s = section_crossline(src, idx, 200)
    fig, ax = plt.subplots()
    assert plot_section(s, ax, xaxis="trace") is ax
    assert ILINES  # silence unused import in some linters
    plt.close(fig)

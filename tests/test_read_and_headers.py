import numpy as np
import pytest
from segyio import TraceField

from ep_seismic.volume import HeaderMap, LatticeIndex, open_segy
from ep_seismic.volume.segy import contiguous_runs, header_hints

from .conftest import ILINES, SAMPLES, XLINES, trace_values

N = len(ILINES) * len(XLINES)


def expected(trace_number: int) -> np.ndarray:
    i, j = divmod(trace_number, len(XLINES))
    return trace_values(i, j)


# -- sequential, run-based reads ---------------------------------------------


@pytest.mark.parametrize(
    ("idx", "max_len", "runs"),
    [
        ([], 10, []),
        ([4], 10, [(0, 1)]),
        ([3, 4, 5], 10, [(0, 3)]),
        ([3, 4, 6, 7, 9], 10, [(0, 2), (2, 4), (4, 5)]),
        ([2, 2, 3], 10, [(0, 1), (1, 3)]),  # a repeat breaks the run
        (list(range(10)), 4, [(0, 4), (4, 8), (8, 10)]),  # cut at the cap
    ],
)
def test_contiguous_runs(idx, max_len, runs) -> None:
    assert contiguous_runs(np.array(idx, dtype=np.int64), max_len) == runs


def test_read_returns_requested_order_with_repeats(make_segy) -> None:
    with open_segy(make_segy()) as src:
        asked = [7, 3, 4, 5, 3, 11, 0, 7]
        data = src.read_traces(asked)
    assert data.shape == (len(asked), len(SAMPLES))
    for row, tn in zip(data, asked, strict=True):
        np.testing.assert_array_equal(row, expected(tn))


def test_read_with_window_and_unsorted_input(make_segy) -> None:
    with open_segy(make_segy()) as src:
        data = src.read_traces([9, 2, 3], sample_slice=slice(1, 4))
    for row, tn in zip(data, [9, 2, 3], strict=True):
        np.testing.assert_array_equal(row, expected(tn)[1:4])


def test_adjacent_traces_are_one_read_and_order_is_ascending(
    make_segy, monkeypatch
) -> None:
    with open_segy(make_segy()) as src:
        calls: list[tuple[int, int]] = []
        real = src._read_run

        def spy(first: int, stop: int):
            calls.append((first, stop))
            return real(first, stop)

        monkeypatch.setattr(src, "_read_run", spy)
        src.read_traces([8, 9, 10, 11, 0, 1, 5])  # scrambled across the file
    assert calls == [(0, 2), (5, 6), (8, 12)]


def test_run_cap_limits_memory_per_read(make_segy, monkeypatch) -> None:
    import ep_seismic.volume.segy as segy_mod

    monkeypatch.setattr(segy_mod, "_RUN_BYTES", len(SAMPLES) * 4 * 3)  # 3 traces
    with open_segy(make_segy()) as src:
        calls: list[tuple[int, int]] = []
        real = src._read_run
        monkeypatch.setattr(
            src, "_read_run", lambda a, b: (calls.append((a, b)), real(a, b))[1]
        )
        data = src.read_traces(list(range(8)))
    assert all(b - a <= 3 for a, b in calls) and len(calls) == 3
    for tn in range(8):
        np.testing.assert_array_equal(data[tn], expected(tn))


# -- header map: validate, remember, invalidate the index, describe -----------


def test_header_map_validation_and_json_roundtrip() -> None:
    hm = HeaderMap(inline=9, crossline=21)
    assert HeaderMap.from_json(hm.to_json()) == hm
    for bad in ({"inline": 0}, {"inline": 238}, {"scalar": 240}, {"cdp_x": 1.5}):
        with pytest.raises(ValueError, match="header byte"):
            HeaderMap(**bad)
    HeaderMap(scalar=239, inline=237)  # last legal starts


def test_unusual_layout_is_flagged_then_fixed_by_a_remembered_map(
    make_segy, tmp_path
) -> None:
    cache = tmp_path / "c"
    path = make_segy(inline_byte=9, crossline_byte=21)  # FieldRecord / CDP bytes
    with open_segy(path, cache_dir=cache) as src:
        assert src.header_map_origin == "default"
        bad = src.index()
        assert not isinstance(bad, LatticeIndex)
        assert any("all zero" in w for w in bad.meta.warnings)
        src.set_header_map(HeaderMap(inline=9, crossline=21))
        assert src.header_map_origin == "user"
        good = src.index()  # cached index built with other bytes must not be reused
        assert isinstance(good, LatticeIndex)
        assert good.inline_range == (ILINES[0], ILINES[-1])
    with open_segy(path, cache_dir=cache) as again:
        assert again.header_map_origin == "remembered"
        assert again.header_map.inline == 9
        assert isinstance(again.index(), LatticeIndex)


def test_header_map_not_remembered_when_asked(make_segy, tmp_path) -> None:
    path, cache = make_segy(), tmp_path / "c"
    with open_segy(path, cache_dir=cache) as src:
        src.set_header_map(HeaderMap(inline=9), remember=False)
    with open_segy(path, cache_dir=cache) as src2:
        assert src2.header_map_origin == "default"


def test_explicit_map_wins_and_corrupt_sidecar_is_ignored(make_segy, tmp_path) -> None:
    path, cache = make_segy(), tmp_path / "c"
    with open_segy(path, cache_dir=cache) as src:
        sidecar = src._header_sidecar()
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text('{"inline": 999}', encoding="utf-8")  # fails validation
        assert src.header_map_origin == "default"
    with open_segy(path, HeaderMap(inline=9), cache_dir=cache) as src2:
        assert (src2.header_map.inline, src2.header_map_origin) == (9, "user")


def test_index_records_the_header_map_it_used(make_segy) -> None:
    with open_segy(make_segy(), HeaderMap()) as src:
        meta = src.index(compact=False).meta
    assert HeaderMap.from_json(meta.header_map) == HeaderMap()


def test_header_hints_parse_real_style_lines() -> None:
    text = (
        "C36 SRC POS X: BYTE 73-76, SRC POS Y: BYTE 77-80\n"
        "C37 REC POS X: BYTE 81-84, REC POS Y: BYTE 85-88\n"
        "C38 MD: BYTE 41-44, TVDSS: BYTE 53-56, CDP: BYTE 5-8\n"
        "C39 POLARITY: RECORDED"
    )
    assert header_hints(text) == [
        ("SRC POS X", 73, 76),
        ("SRC POS Y", 77, 80),
        ("REC POS X", 81, 84),
        ("REC POS Y", 85, 88),
        ("MD", 41, 44),
        ("TVDSS", 53, 56),
        ("CDP", 5, 8),
    ]
    assert header_hints("nothing useful here") == []


def test_describe_headers_shows_values_and_claims(make_segy) -> None:
    path = make_segy(text="CDP X: BYTE 181-184")
    with open_segy(path) as src:
        info = src.inspect_headers()
        assert info["traces"] == [0, N // 2, N - 1]
        assert info["fields"]["inline"] == {
            "byte": TraceField.INLINE_3D,
            "values": [100, 101, 102],
        }
        text = src.describe_headers([0])
    assert "Header map in use: default" in text
    assert "inline" in text and "byte 189" in text
    assert "Text header says (unverified)" in text and "bytes 181-184" in text

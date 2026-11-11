"""Synthetic SEG-Y factory. Tests never use real or licensed data."""

from __future__ import annotations

import numpy as np
import pytest
import segyio
from segyio import TraceField

ILINES = [100, 101, 102]
XLINES = [200, 201, 202, 203]
SAMPLES = np.arange(0, 40, 4.0)  # 10 samples, 4 ms


def trace_values(i: int, j: int, offset_index: int = 0) -> np.ndarray:
    """Distinct, recomputable values per grid cell (indices, not line numbers)."""
    base = i * 1000 + j * 10 + offset_index * 100000
    return (base + np.arange(len(SAMPLES))).astype(np.float32)


def true_xy(i: int, j: int, theta: float = 0.0) -> tuple[float, float]:
    """Bin centre: 25 m along inline index, 12.5 m along crossline index."""
    u, v = 25.0 * i, 12.5 * j
    c, s = np.cos(theta), np.sin(theta)
    return 1000.0 + u * c - v * s, 5000.0 + u * s + v * c


@pytest.fixture
def make_segy(tmp_path):
    """Build a tiny SEG-Y.

    ``ilines`` / ``xlines`` give the line numbers (index i / j are positions in
    these lists). ``order`` is which varies fastest in the file. ``drop`` removes
    grid cells; ``jitter`` moves one cell's coordinates by that many metres in X;
    ``offsets`` writes one trace per offset at every cell. Coordinates are stored
    as integers with the given ``scalar``.
    """

    def _make(
        name: str = "synth.sgy",
        *,
        ilines: list[int] = ILINES,
        xlines: list[int] = XLINES,
        order: str = "inline_major",
        scalar: int = -100,
        use_cdp: bool = True,
        zero_lines: bool = False,
        endian: str = "big",
        theta: float = 0.0,
        drop: tuple[tuple[int, int], ...] = (),
        jitter: tuple[int, int, float] | None = None,
        offsets: tuple[int, ...] = (0,),
        text: str = "",
        folder: str = "",
        inline_byte: int = TraceField.INLINE_3D,
        crossline_byte: int = TraceField.CROSSLINE_3D,
        delay: int = 0,
        time_scalar: int = 0,
    ):
        folder_path = tmp_path / folder if folder else tmp_path
        folder_path.mkdir(parents=True, exist_ok=True)
        path = folder_path / name
        cells = [(i, j) for i in range(len(ilines)) for j in range(len(xlines))]
        if order == "xline_major":
            cells.sort(key=lambda ij: (ij[1], ij[0]))
        cells = [c for c in cells if c not in drop]
        spec = segyio.spec()
        spec.format = 5
        spec.endian = endian
        spec.samples = SAMPLES
        spec.tracecount = len(cells) * len(offsets)
        with segyio.create(path, spec) as f:
            f.bin[segyio.BinField.MeasurementSystem] = 1
            if text:
                f.text[0] = segyio.tools.create_text_header({1: text})
            n = 0
            for i, j in cells:
                x_true, y_true = true_xy(i, j, theta)
                if jitter and (i, j) == jitter[:2]:
                    x_true += jitter[2]
                for k, off in enumerate(offsets):
                    if scalar < 0:
                        xi, yi = round(x_true * -scalar), round(y_true * -scalar)
                    elif scalar > 0:
                        xi, yi = round(x_true / scalar), round(y_true / scalar)
                    else:
                        xi, yi = round(x_true), round(y_true)
                    h = {
                        inline_byte: 0 if zero_lines else ilines[i],
                        crossline_byte: 0 if zero_lines else xlines[j],
                        TraceField.offset: off,
                        TraceField.SourceGroupScalar: scalar,
                        TraceField.CoordinateUnits: 1,
                        TraceField.DelayRecordingTime: delay,
                        TraceField.ScalarTraceHeader: time_scalar,
                    }
                    if use_cdp:
                        h[TraceField.CDP_X], h[TraceField.CDP_Y] = xi, yi
                    else:
                        h[TraceField.SourceX], h[TraceField.SourceY] = xi, yi
                    f.header[n] = h
                    f.trace[n] = trace_values(i, j, k)
                    n += 1
        return path

    return _make

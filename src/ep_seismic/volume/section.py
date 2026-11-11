"""Small 2D sections cut from a large volume: one line, or an arbitrary path.

Only the traces on the section are read. Every section carries the vertical
domain and quantity (a section cannot be made until they are settled) and notes
saying how it was made, so a saved or plotted section stays honest on its own.
Arbitrary paths use the nearest trace to each sample point: no interpolation,
no resampling of the volume.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ep_seismic.volume.domain import SampleDomain
from ep_seismic.volume.index import GeometryIndex
from ep_seismic.volume.source import DEFAULT_MAX_BYTES, SeismicSource

_FORMAT_VERSION = 1


@dataclass(frozen=True)
class Section:
    """Traces along a line, with their positions and what the samples mean.

    ``data`` is ``(n_traces, n_samples)``. ``distance`` is along the section from
    its first trace, in the file's coordinate units (SEG-Y does not say whether
    these are metres or feet). ``samples`` is the vertical axis in the unit of
    ``domain.vertical_unit``.
    """

    data: np.ndarray
    samples: np.ndarray
    trace_numbers: np.ndarray
    x: np.ndarray
    y: np.ndarray
    inline: np.ndarray
    crossline: np.ndarray
    distance: np.ndarray
    domain: SampleDomain
    kind: str
    source_name: str
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        n = self.data.shape[0]
        for name in ("trace_numbers", "x", "y", "inline", "crossline", "distance"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have one value per trace ({n})")
        if self.samples.shape != (self.data.shape[1],):
            raise ValueError("samples must match the trace length")

    @property
    def n_traces(self) -> int:
        return self.data.shape[0]

    def caption(self) -> str:
        """One line saying what this is and how it was decided."""
        return f"{self.source_name}: {self.kind}. {self.domain.describe()}"

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Save to a small ``.npz`` (no pickles). This is the thing worth keeping."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "format_version": _FORMAT_VERSION,
            "domain": asdict(self.domain),
            "kind": self.kind,
            "source_name": self.source_name,
            "notes": list(self.notes),
        }
        np.savez_compressed(
            path,
            data=self.data,
            samples=self.samples,
            trace_numbers=self.trace_numbers,
            x=self.x,
            y=self.y,
            inline=self.inline,
            crossline=self.crossline,
            distance=self.distance,
            meta=np.array(json.dumps(meta)),
        )
        return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Section:
        """Read a section written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("format_version") != _FORMAT_VERSION:
                raise ValueError("unsupported section format version")
            return cls(
                data=z["data"],
                samples=z["samples"],
                trace_numbers=z["trace_numbers"],
                x=z["x"],
                y=z["y"],
                inline=z["inline"],
                crossline=z["crossline"],
                distance=z["distance"],
                domain=SampleDomain(**meta["domain"]),
                kind=meta["kind"],
                source_name=meta["source_name"],
                notes=tuple(meta["notes"]),
            )


def _window_slice(samples: np.ndarray, window: tuple[float, float] | None) -> slice:
    """Sample slice for a (start, stop) window in the sample axis unit, inclusive."""
    if window is None:
        return slice(None)
    lo, hi = sorted(window)
    first = int(np.searchsorted(samples, lo, side="left"))
    last = int(np.searchsorted(samples, hi, side="right"))
    if last <= first:
        raise ValueError(
            f"window {window} is outside the samples {samples[0]}..{samples[-1]}"
        )
    return slice(first, last)


def _build(
    src: SeismicSource,
    idx: GeometryIndex,
    trace_numbers: np.ndarray,
    distance: np.ndarray,
    *,
    kind: str,
    window: tuple[float, float] | None,
    domain: SampleDomain | None,
    max_bytes: int,
    notes: Sequence[str],
) -> Section:
    dom = domain if domain is not None else src.require_domain()
    sl = _window_slice(src.samples, window)
    data = src.read_traces(trace_numbers, sample_slice=sl, max_bytes=max_bytes)
    x, y = idx.xy(trace_numbers)
    il, xl = idx.ilxl(trace_numbers)
    all_notes = (*idx.meta.warnings, *notes)
    return Section(
        data=data,
        samples=np.asarray(src.samples[sl]),
        trace_numbers=np.asarray(trace_numbers, dtype=np.int64),
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        inline=np.asarray(il),
        crossline=np.asarray(xl),
        distance=np.asarray(distance, dtype=np.float64),
        domain=dom,
        kind=kind,
        source_name=Path(idx.meta.source_path).name,
        notes=tuple(all_notes),
    )


def _along(idx: GeometryIndex, tn: np.ndarray) -> np.ndarray:
    """Distance along consecutive traces, from the first, in coordinate units."""
    x, y = idx.xy(tn)
    return np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))


def _restrict(
    idx: GeometryIndex,
    tn: np.ndarray,
    which: str,
    bounds: tuple[int, int] | None,
) -> np.ndarray:
    if bounds is None:
        return tn
    lo, hi = sorted(bounds)
    il, xl = idx.ilxl(tn)
    values = xl if which == "crossline" else il
    keep = tn[(values >= lo) & (values <= hi)]
    if keep.size == 0:
        raise ValueError(f"no traces within {which}s {lo}..{hi}")
    return keep


def section_inline(
    src: SeismicSource,
    idx: GeometryIndex,
    inline: int,
    *,
    crossline_range: tuple[int, int] | None = None,
    window: tuple[float, float] | None = None,
    domain: SampleDomain | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Section:
    """Section along one inline, optionally trimmed in crossline and in time/depth.

    ``window`` is ``(start, stop)`` in the sample axis unit (ms or m). The domain
    must be settled (``src.set_domain`` or pass ``domain``).

    Raises
    ------
    DomainUnspecified
        If the vertical domain or quantity is not known.
    ValueError
        If the line, range or window is not in the data, or the read is too big.
    """
    tn = _restrict(idx, idx.select(inline=inline), "crossline", crossline_range)
    return _build(
        src,
        idx,
        tn,
        _along(idx, tn),
        kind=f"inline {inline}",
        window=window,
        domain=domain,
        max_bytes=max_bytes,
        notes=(),
    )


def section_crossline(
    src: SeismicSource,
    idx: GeometryIndex,
    crossline: int,
    *,
    inline_range: tuple[int, int] | None = None,
    window: tuple[float, float] | None = None,
    domain: SampleDomain | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Section:
    """Section along one crossline; see :func:`section_inline`."""
    tn = _restrict(idx, idx.select(crossline=crossline), "inline", inline_range)
    return _build(
        src,
        idx,
        tn,
        _along(idx, tn),
        kind=f"crossline {crossline}",
        window=window,
        domain=domain,
        max_bytes=max_bytes,
        notes=(),
    )


def section_path(
    src: SeismicSource,
    idx: GeometryIndex,
    vertices: Sequence[Sequence[float]] | np.ndarray,
    *,
    vertices_kind: str = "xy",
    spacing: float | None = None,
    max_distance: float | None = None,
    window: tuple[float, float] | None = None,
    domain: SampleDomain | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Section:
    """Section along an arbitrary polyline through the survey.

    ``vertices`` are ``(x, y)`` in the file's coordinates, or ``(inline,
    crossline)`` with ``vertices_kind="ilxl"`` (fractional values are fine on a
    regular grid). Sample points are placed every ``spacing`` (default: one bin)
    along the polyline and each takes the NEAREST trace; repeats are dropped.
    Points outside the survey are dropped and counted in the notes. There is no
    interpolation.

    Raises
    ------
    DomainUnspecified
        If the vertical domain or quantity is not known.
    ValueError
        If the path is invalid, lies wholly outside the survey, or is too big.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 2 or v.shape[0] < 2:
        raise ValueError(
            "vertices must be at least two (x, y) or (inline, xline) pairs"
        )
    if vertices_kind == "ilxl":
        vx, vy = idx.xy_of_ilxl(v[:, 0], v[:, 1])
    elif vertices_kind == "xy":
        vx, vy = v[:, 0], v[:, 1]
    else:
        raise ValueError("vertices_kind must be 'xy' or 'ilxl'")
    seg = np.hypot(np.diff(vx), np.diff(vy))
    total = float(seg.sum())
    if total == 0:
        raise ValueError("path has zero length")
    step = float(spacing) if spacing is not None else idx.bin_size
    if step <= 0:
        raise ValueError("spacing must be positive")
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    s = np.unique(np.append(np.arange(0.0, total, step), total))
    px, py = np.interp(s, cum, vx), np.interp(s, cum, vy)
    tn, _ = idx.nearest(px, py, max_distance=max_distance)

    outside = tn < 0
    notes = ["nearest trace to each point on the path; no interpolation"]
    if outside.all():
        raise ValueError("the path lies entirely outside the survey")
    if outside.any():
        notes.append(f"{int(outside.sum())} of {s.size} path points outside the survey")
    s, tn = s[~outside], tn[~outside]
    first = np.concatenate(([True], tn[1:] != tn[:-1]))
    s, tn = s[first], tn[first]
    return _build(
        src,
        idx,
        tn,
        s - s[0],
        kind=f"arbitrary path ({v.shape[0]} vertices, {total:.0f} long)",
        window=window,
        domain=domain,
        max_bytes=max_bytes,
        notes=notes,
    )

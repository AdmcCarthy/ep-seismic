"""Geometry indexes: small sidecars describing where every trace sits.

Two forms share one interface (:class:`GeometryIndex`):

* :class:`LatticeIndex` - a regular 3D grid (every inline x crossline position has
  exactly one trace, in order). A few numbers describe the whole survey, however
  many traces it has.
* :class:`TraceIndex` - one row per trace (inline, crossline, offset, X, Y). Used
  for irregular surveys, 2D lines and gathers. About 28 bytes per trace on disk.

Both are stored outside the repository and tied to the source file by size and
modification time.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.spatial import cKDTree

CACHE_ENV = "SEISMIC_LIB_CACHE"
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class IndexMeta:
    """What the index was built from, and the choices made while building it.

    Sample start and step are in the unit of the sample axis, which the SEG-Y
    headers do not state as time or depth: ms if time, m (or ft) if depth.
    ``measurement_system`` and ``coordinate_units`` are the raw SEG-Y codes
    (measurement system: 1 metres, 2 feet; coordinate units: 1 length, 2 arc
    seconds, 3 decimal degrees, 4 DMS), 0 when unset.
    """

    source_path: str
    source_size: int
    source_mtime_ns: int
    n_traces: int
    n_samples: int
    sample_start: float
    sample_step: float
    xy_fields: str
    measurement_system: int
    coordinate_units: int
    warnings: tuple[str, ...] = field(default_factory=tuple)
    header_map: str = ""  # JSON of the header bytes used; a changed map voids the index


class GeometryIndex(Protocol):
    """What sections need from an index, whichever form it takes."""

    meta: IndexMeta

    @property
    def n_traces(self) -> int: ...

    @property
    def bin_size(self) -> float:
        """Typical distance between neighbouring traces, in coordinate units."""
        ...

    @property
    def inline_range(self) -> tuple[int, int]: ...

    @property
    def crossline_range(self) -> tuple[int, int]: ...

    def select(
        self, *, inline: int | None = None, crossline: int | None = None
    ) -> np.ndarray:
        """Trace numbers on one inline or crossline, ordered along the line."""
        ...

    def locate(self, inline: np.ndarray, crossline: np.ndarray) -> np.ndarray:
        """Trace number at each (inline, crossline), -1 where there is none."""
        ...

    def xy(self, trace_numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def ilxl(self, trace_numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inline and crossline numbers of the given traces."""
        ...

    def xy_of_ilxl(
        self, inline: np.ndarray, crossline: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of (inline, crossline), which may be fractional on a grid."""
        ...

    def nearest(
        self, x: np.ndarray, y: np.ndarray, max_distance: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nearest trace to each point: ``(trace_numbers, distances)``.

        Trace number is -1 where the point is outside the survey (or farther
        than ``max_distance``).
        """
        ...

    def is_current_for(self, path: str | os.PathLike[str]) -> bool: ...

    def save(self, path: str | os.PathLike[str]) -> Path: ...


def _is_current(meta: IndexMeta, path: str | os.PathLike[str]) -> bool:
    st = Path(path).stat()
    return st.st_size == meta.source_size and st.st_mtime_ns == meta.source_mtime_ns


def _write_npz(path: str | os.PathLike[str], arrays: dict, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            **arrays,
            meta=np.array(json.dumps({"format_version": _FORMAT_VERSION, **meta})),
        )
    os.replace(tmp, path)
    return path


def _no_traces(inline: int | None, crossline: int | None) -> ValueError:
    which = f"inline {inline}" if inline is not None else f"crossline {crossline}"
    return ValueError(f"no traces on {which}")


def _grid_position(value: int, v0: int, step: int, n: int) -> int | None:
    """Grid index of a line number, or None if it is off the grid."""
    q, r = divmod(value - v0, step)
    return int(q) if r == 0 and 0 <= q < n else None


def _check_one_of(inline: int | None, crossline: int | None) -> None:
    if (inline is None) == (crossline is None):
        raise ValueError("give exactly one of inline or crossline")


@dataclass(frozen=True)
class TraceIndex:
    """Per-trace geometry. Arrays are aligned: row i describes trace i of the file."""

    inline: np.ndarray
    crossline: np.ndarray
    offset: np.ndarray
    x: np.ndarray
    y: np.ndarray
    meta: IndexMeta

    def __post_init__(self) -> None:
        n = self.meta.n_traces
        for name in ("inline", "crossline", "offset", "x", "y"):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"{name} has shape {arr.shape}, expected ({n},)")

    @property
    def n_traces(self) -> int:
        return self.meta.n_traces

    @property
    def inline_range(self) -> tuple[int, int]:
        return int(self.inline.min()), int(self.inline.max())

    @property
    def crossline_range(self) -> tuple[int, int]:
        return int(self.crossline.min()), int(self.crossline.max())

    def select(
        self, *, inline: int | None = None, crossline: int | None = None
    ) -> np.ndarray:
        """Trace numbers on one inline or one crossline, ordered along the line.

        Exactly one argument must be given. Traces are ordered by crossline (for
        an inline) or inline (for a crossline), then by offset, so several
        offsets at one position stay together.

        Raises
        ------
        ValueError
            If both or neither argument is given, or nothing matches.
        """
        _check_one_of(inline, crossline)
        if inline is not None:
            hit = np.flatnonzero(self.inline == inline)
            along = self.crossline
        else:
            hit = np.flatnonzero(self.crossline == crossline)
            along = self.inline
        if hit.size == 0:
            raise _no_traces(inline, crossline)
        order = np.lexsort((self.offset[hit], along[hit]))
        return hit[order]

    @cached_property
    def _sorted_keys(self) -> tuple[np.ndarray, np.ndarray]:
        key = (self.inline.astype(np.int64) << 32) | (
            self.crossline.astype(np.int64) & 0xFFFFFFFF
        )
        order = np.argsort(key, kind="stable")  # stable: lowest trace number first
        return key[order], order

    def locate(self, inline: np.ndarray, crossline: np.ndarray) -> np.ndarray:
        """Trace number at each (inline, crossline), -1 where there is none.

        With several offsets at one position, the lowest trace number is returned.
        """
        il, xl = np.broadcast_arrays(np.asarray(inline), np.asarray(crossline))
        keys, order = self._sorted_keys
        if not (np.all(il == np.rint(il)) and np.all(xl == np.rint(xl))):
            raise ValueError("inline and crossline must be whole numbers here")
        q = (il.astype(np.int64) << 32) | (xl.astype(np.int64) & 0xFFFFFFFF)
        pos = np.minimum(np.searchsorted(keys, q), keys.size - 1)
        return np.where(keys[pos] == q, order[pos], -1)

    def xy(self, trace_numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tn = np.asarray(trace_numbers)
        return self.x[tn], self.y[tn]

    def ilxl(self, trace_numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inline and crossline numbers of the given traces."""
        tn = np.asarray(trace_numbers)
        return self.inline[tn], self.crossline[tn]

    def xy_of_ilxl(
        self, inline: np.ndarray, crossline: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates at whole-number (inline, crossline) positions.

        Raises
        ------
        ValueError
            If a position has no trace, or the numbers are fractional.
        """
        tn = self.locate(inline, crossline)
        if (tn < 0).any():
            raise ValueError("some (inline, crossline) positions have no trace")
        return self.xy(tn)

    @cached_property
    def _tree(self) -> cKDTree:
        return cKDTree(np.column_stack((self.x, self.y)))

    @cached_property
    def bin_size(self) -> float:
        """Estimated trace spacing: median nearest distinct-position distance."""
        n = self.n_traces
        pick = np.linspace(0, n - 1, min(n, 2000)).astype(np.int64)
        k = min(4, n)
        if k < 2:
            return 0.0
        d, _ = self._tree.query(np.column_stack((self.x[pick], self.y[pick])), k=k)
        d = np.where(d > 0, d, np.inf)[:, 1:].min(axis=1)
        d = d[np.isfinite(d)]
        return float(np.median(d)) if d.size else 0.0

    def nearest(
        self, x: np.ndarray, y: np.ndarray, max_distance: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nearest trace to each point; -1 beyond ``max_distance`` (default: 1 bin)."""
        limit = self.bin_size if max_distance is None else max_distance
        d, i = self._tree.query(np.column_stack((np.ravel(x), np.ravel(y))))
        return np.where(d <= limit, i, -1).astype(np.int64), d

    def is_current_for(self, path: str | os.PathLike[str]) -> bool:
        """True if the file at ``path`` has the size and mtime this index recorded."""
        return _is_current(self.meta, path)

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the index to a compressed ``.npz`` (no pickles)."""
        arrays = {
            k: getattr(self, k) for k in ("inline", "crossline", "offset", "x", "y")
        }
        return _write_npz(path, arrays, {"kind": "full", **asdict(self.meta)})

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> TraceIndex:
        """Read a per-trace index written by :meth:`save`.

        Raises
        ------
        ValueError
            If the file is a different kind of index or an incompatible version.
        """
        idx = load_index(path)
        if not isinstance(idx, cls):
            raise ValueError(f"{path} holds a {type(idx).__name__}, not a TraceIndex")
        return idx


@dataclass(frozen=True)
class LatticeIndex:
    """A regular 3D grid described by a few numbers.

    Trace ``k`` sits at grid cell ``(i, j)`` with ``inline = il0 + il_step * i`` and
    ``crossline = xl0 + xl_step * j``. ``order`` says which varies fastest in the
    file: ``"inline_major"`` means crosslines are contiguous within an inline
    (``k = i * n_xl + j``); ``"xline_major"`` the reverse (``k = j * n_il + i``).
    Coordinates are affine: ``p = p0 + i * d_il + j * d_xl``. The builder checks
    every trace against this model and records the worst coordinate residual.
    """

    meta: IndexMeta
    order: str
    n_il: int
    n_xl: int
    il0: int
    il_step: int
    xl0: int
    xl_step: int
    p0: tuple[float, float]
    d_il: tuple[float, float]
    d_xl: tuple[float, float]
    offset: int
    max_xy_residual: float

    def __post_init__(self) -> None:
        if self.order not in ("inline_major", "xline_major"):
            raise ValueError(f"unknown trace order {self.order!r}")
        if self.n_il * self.n_xl != self.meta.n_traces:
            raise ValueError("grid size does not match the trace count")
        if self.il_step == 0 or self.xl_step == 0:
            raise ValueError("line steps must be non-zero")
        if self.n_il < 2 or self.n_xl < 2:
            raise ValueError("a lattice needs at least 2 inlines and 2 crosslines")

    @property
    def n_traces(self) -> int:
        return self.meta.n_traces

    @property
    def inline_values(self) -> np.ndarray:
        return self.il0 + self.il_step * np.arange(self.n_il)

    @property
    def crossline_values(self) -> np.ndarray:
        return self.xl0 + self.xl_step * np.arange(self.n_xl)

    @property
    def inline_range(self) -> tuple[int, int]:
        v = self.inline_values
        return int(v.min()), int(v.max())

    @property
    def crossline_range(self) -> tuple[int, int]:
        v = self.crossline_values
        return int(v.min()), int(v.max())

    @property
    def bin_size(self) -> float:
        """Smaller of the inline and crossline spacings, in coordinate units."""
        return float(min(np.hypot(*self.d_il), np.hypot(*self.d_xl)))

    @cached_property
    def _inverse(self) -> np.ndarray:
        m = np.array([[self.d_il[0], self.d_xl[0]], [self.d_il[1], self.d_xl[1]]])
        if abs(np.linalg.det(m)) < 1e-12:
            raise ValueError("inline and crossline directions are not independent")
        return np.linalg.inv(m)

    def _trace_number(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        if self.order == "inline_major":
            return i * self.n_xl + j
        return j * self.n_il + i

    def _ij(self, trace_numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tn = np.asarray(trace_numbers, dtype=np.int64)
        if self.order == "inline_major":
            i, j = np.divmod(tn, self.n_xl)
        else:
            j, i = np.divmod(tn, self.n_il)
        return i, j

    def select(
        self, *, inline: int | None = None, crossline: int | None = None
    ) -> np.ndarray:
        """Trace numbers on one inline or one crossline, ordered along the line.

        Ordered by ascending crossline (for an inline) or inline (for a crossline)
        number, whatever the order in the file.

        Raises
        ------
        ValueError
            If both or neither argument is given, or the line is not in the grid.
        """
        _check_one_of(inline, crossline)
        if inline is not None:
            i = _grid_position(inline, self.il0, self.il_step, self.n_il)
            if i is None:
                raise _no_traces(inline, crossline)
            j = np.arange(self.n_xl, dtype=np.int64)
            if self.xl_step < 0:
                j = j[::-1]  # ascending crossline number, whatever the file order
            return self._trace_number(np.full_like(j, i), j)
        j = _grid_position(crossline, self.xl0, self.xl_step, self.n_xl)
        if j is None:
            raise _no_traces(inline, crossline)
        i = np.arange(self.n_il, dtype=np.int64)
        if self.il_step < 0:
            i = i[::-1]
        return self._trace_number(i, np.full_like(i, j))

    def locate(self, inline: np.ndarray, crossline: np.ndarray) -> np.ndarray:
        """Trace number at each (inline, crossline), -1 where there is none."""
        il, xl = np.broadcast_arrays(np.asarray(inline), np.asarray(crossline))
        fi = (il - self.il0) / self.il_step
        fj = (xl - self.xl0) / self.xl_step
        i, j = np.rint(fi), np.rint(fj)
        ok = (
            (fi == i)
            & (fj == j)
            & (i >= 0)
            & (i < self.n_il)
            & (j >= 0)
            & (j < self.n_xl)
        )
        tn = self._trace_number(i.astype(np.int64), j.astype(np.int64))
        return np.where(ok, tn, -1)

    def xy_of_ilxl(
        self, inline: np.ndarray, crossline: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of (inline, crossline); fractional line numbers are fine."""
        fi = (np.asarray(inline, dtype=np.float64) - self.il0) / self.il_step
        fj = (np.asarray(crossline, dtype=np.float64) - self.xl0) / self.xl_step
        return self._xy_of_ij(fi, fj)

    def _xy_of_ij(self, i: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.p0[0] + i * self.d_il[0] + j * self.d_xl[0]
        y = self.p0[1] + i * self.d_il[1] + j * self.d_xl[1]
        return x, y

    def xy(self, trace_numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        i, j = self._ij(trace_numbers)
        return self._xy_of_ij(i, j)

    def ilxl(self, trace_numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inline and crossline numbers of the given traces."""
        i, j = self._ij(trace_numbers)
        return self.il0 + self.il_step * i, self.xl0 + self.xl_step * j

    def nearest(
        self, x: np.ndarray, y: np.ndarray, max_distance: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nearest trace to each point; -1 outside the survey outline.

        The outline is the grid plus half a bin on every side. ``max_distance``
        additionally rejects points farther than that from the chosen trace.
        """
        dx = np.ravel(x).astype(np.float64) - self.p0[0]
        dy = np.ravel(y).astype(np.float64) - self.p0[1]
        fij = self._inverse @ np.vstack((dx, dy))
        i, j = np.rint(fij[0]).astype(np.int64), np.rint(fij[1]).astype(np.int64)
        inside = (i >= 0) & (i < self.n_il) & (j >= 0) & (j < self.n_xl)
        ic, jc = np.clip(i, 0, self.n_il - 1), np.clip(j, 0, self.n_xl - 1)
        cx, cy = self._xy_of_ij(ic, jc)
        dist = np.hypot(cx - np.ravel(x), cy - np.ravel(y))
        if max_distance is not None:
            inside &= dist <= max_distance
        return np.where(inside, self._trace_number(ic, jc), -1), dist

    def is_current_for(self, path: str | os.PathLike[str]) -> bool:
        """True if the file at ``path`` has the size and mtime this index recorded."""
        return _is_current(self.meta, path)

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the lattice description to a tiny ``.npz`` (no pickles)."""
        params = asdict(self)
        meta = params.pop("meta")
        return _write_npz(path, {}, {"kind": "lattice", **meta, "lattice": params})

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> LatticeIndex:
        """Read a lattice index written by :meth:`save`."""
        idx = load_index(path)
        if not isinstance(idx, cls):
            raise ValueError(f"{path} holds a {type(idx).__name__}, not a LatticeIndex")
        return idx


def load_index(path: str | os.PathLike[str]) -> TraceIndex | LatticeIndex:
    """Read whichever kind of index is stored at ``path``.

    Raises
    ------
    ValueError
        If the file was written by an incompatible version or has an unknown kind.
    """
    with np.load(path, allow_pickle=False) as z:
        raw = json.loads(str(z["meta"]))
        version = raw.pop("format_version", None)
        if version != _FORMAT_VERSION:
            raise ValueError(f"unsupported index format version {version!r}")
        kind = raw.pop("kind", "full")
        params = raw.pop("lattice", None)
        raw["warnings"] = tuple(raw["warnings"])
        meta = IndexMeta(**raw)
        if kind == "full":
            return TraceIndex(
                inline=z["inline"],
                crossline=z["crossline"],
                offset=z["offset"],
                x=z["x"],
                y=z["y"],
                meta=meta,
            )
        if kind == "lattice" and params is not None:
            for k in ("p0", "d_il", "d_xl"):
                params[k] = tuple(params[k])
            return LatticeIndex(meta=meta, **params)
    raise ValueError(f"unknown index kind {kind!r}")


def cache_dir(override: str | os.PathLike[str] | None = None) -> Path:
    """Folder for index sidecars: argument, else ``SEISMIC_LIB_CACHE``, else home."""
    if override is not None:
        return Path(override)
    env = os.environ.get(CACHE_ENV)
    if env:
        return Path(env)
    return Path.home() / ".seismic_lib" / "index"


def index_path(
    source: str | os.PathLike[str], directory: str | os.PathLike[str] | None = None
) -> Path:
    """Sidecar path for a source file: its name plus a hash of its full path."""
    resolved = Path(source).resolve()
    tag = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:10]
    return cache_dir(directory) / f"{resolved.stem}-{tag}.npz"

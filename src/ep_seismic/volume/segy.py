"""SEG-Y source, read in place with segyio."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import segyio
from segyio import BinField, TraceField

from ep_seismic.volume.domain import (
    DomainUnspecified,
    SampleDomain,
    detect_domain,
    domain_from_detection,
    domain_from_json,
    domain_to_json,
    unspecified_message,
)
from ep_seismic.volume.index import (
    GeometryIndex,
    IndexMeta,
    LatticeIndex,
    TraceIndex,
    index_path,
    load_index,
)
from ep_seismic.volume.source import DEFAULT_MAX_BYTES

_FILE_HEADER_BYTES = 3600  # textual (3200) + binary (400); extended headers add 3200
_TRACE_HEADER_BYTES = 240
# SEG-Y sample formats: 1 IBM float, 2 int32, 3 int16, 5 IEEE float, 8 int8
_SAMPLE_BYTES = {1: 4, 2: 4, 3: 2, 5: 4, 8: 1}
_RUN_BYTES = 64 * 1024**2  # most memory one run of adjacent traces may take


@dataclass(frozen=True)
class HeaderMap:
    """Trace header bytes to read. Defaults follow SEG-Y rev 1 post-stack practice.

    Each value is the 1-based SEG-Y trace header byte where a field starts (4-byte
    integers, except ``scalar`` which is 2 bytes). X and Y are tried in this order:
    ``cdp_x/cdp_y``, then ``source_x/source_y`` if the CDP pair is all zero. The
    choice is recorded on the index. Files that put these fields elsewhere need a
    corrected map: see :meth:`SegySource.set_header_map`.
    """

    inline: int = TraceField.INLINE_3D
    crossline: int = TraceField.CROSSLINE_3D
    offset: int = TraceField.offset
    cdp_x: int = TraceField.CDP_X
    cdp_y: int = TraceField.CDP_Y
    source_x: int = TraceField.SourceX
    source_y: int = TraceField.SourceY
    scalar: int = TraceField.SourceGroupScalar

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            width = 2 if name == "scalar" else 4
            if (
                not isinstance(value, int)
                or not 1 <= value <= _TRACE_HEADER_BYTES - width + 1
            ):
                raise ValueError(
                    f"header byte for {name} must be a whole number from 1 to "
                    f"{_TRACE_HEADER_BYTES - width + 1}, got {value!r}"
                )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> HeaderMap:
        return cls(**json.loads(text))


def apply_scalar(values: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    """Apply the SEG-Y coordinate scalar per trace.

    Positive: multiply. Negative: divide by the absolute value. Zero: as 1.
    """
    s = scalar.astype(np.float64)
    factor = np.ones_like(s)
    np.copyto(factor, s, where=s > 0)
    np.divide(1.0, np.abs(s), out=factor, where=s < 0)
    return values.astype(np.float64) * factor


_TIME_SCALARS = (1, 10, 100, 1000, 10000)  # the values SEG-Y rev 1 allows
_FIRST_SAMPLE = re.compile(
    r"FIRST\s+SAMPLE\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*(MS|M|FT)?\b", re.IGNORECASE
)


def sample_axis(
    n_samples: int,
    step: float,
    delay: int,
    time_scalar: int,
    text_header: str = "",
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Vertical axis of the traces, built from the headers rather than left to segyio.

    ``step`` is the sample interval in ms (a depth file keeps its step in the same
    field). ``delay`` is the delay recording time of the first trace (bytes 109-110)
    and ``time_scalar`` the scalar for the time fields (bytes 215-216), which SEG-Y
    rev 1 restricts to 1, +/-10, +/-100, +/-1000 and +/-10000: positive multiplies,
    negative divides. Any other value is reported and not applied. segyio multiplies
    the delay by the scalar's absolute value whatever it is; the Volve ST10010 stack
    carries 3 there, which turns a 4 ms delay into 12 ms and puts the whole axis 8 ms
    late. A note is also added when the text header states a FIRST SAMPLE that
    disagrees with the axis; the trace header still wins.

    Returns the axis and the notes, which go on the index as warnings.
    """
    notes: list[str] = []
    start = float(delay)
    if time_scalar in (0, 1):
        pass
    elif abs(time_scalar) in _TIME_SCALARS:
        start = delay * time_scalar if time_scalar > 0 else delay / abs(time_scalar)
    elif delay:
        notes.append(
            f"trace-header time scalar {time_scalar} (bytes 215-216) is not a SEG-Y "
            f"value and was not applied; the first sample is the delay recording "
            f"time {delay} as stored (segyio would make it {delay * abs(time_scalar)})"
        )
    m = _FIRST_SAMPLE.search(text_header)
    if m and abs(float(m.group(1)) - start) > 1e-6:
        unit = f" {m.group(2).upper()}" if m.group(2) else ""
        notes.append(
            f"text header states FIRST SAMPLE {m.group(1)}{unit} but the trace "
            f"header gives {start:g}; the trace header was used"
        )
    return start + np.arange(n_samples) * float(step), tuple(notes)


def _decode_text_header(raw: bytes) -> str:
    """Decode the 3200-byte textual header, which is EBCDIC or ASCII."""
    as_ascii = raw.decode("ascii", errors="replace")
    as_ebcdic = raw.decode("cp500", errors="replace")
    pick = (
        as_ascii
        if sum(c.isalpha() for c in as_ascii) >= sum(c.isalpha() for c in as_ebcdic)
        else as_ebcdic
    )
    return "\n".join(pick[i : i + 80] for i in range(0, len(pick), 80))


def contiguous_runs(sorted_idx: np.ndarray, max_len: int) -> list[tuple[int, int]]:
    """Split a sorted index array into runs of consecutive trace numbers.

    Returns ``(start, stop)`` positions into ``sorted_idx`` (stop exclusive). A run
    breaks at any gap or repeat, and is cut so none is longer than ``max_len``.
    Reading each run in one call keeps disk access sequential.
    """
    n = sorted_idx.size
    if n == 0:
        return []
    breaks = np.flatnonzero(np.diff(sorted_idx) != 1) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [n]))
    runs: list[tuple[int, int]] = []
    for a, b in zip(starts.tolist(), stops.tolist(), strict=True):
        runs.extend((p, min(p + max_len, b)) for p in range(a, b, max_len))
    return runs


_CARD_NUMBER = re.compile(r"^\s*C\s?\d{1,2}(?=\s)")
_BYTE_HINT = re.compile(
    r"([A-Za-z][A-Za-z0-9 /_.\-]*?)\s*[:=]\s*BYTES?\s*(\d+)\s*(?:-|TO)\s*(\d+)",
    re.IGNORECASE,
)


def header_hints(text_header: str) -> list[tuple[str, int, int]]:
    """Byte positions the text header says it uses, as ``(label, first, last)``.

    Finds statements like ``SRC POS X: BYTE 73-76`` or ``CDP: BYTE 5-8``. These are
    what the file's writer says, not verified facts, and are never applied for you:
    read them, check the values with :meth:`SegySource.describe_headers`, then build
    a :class:`HeaderMap`.
    """
    found = []
    for line in text_header.splitlines():
        line = _CARD_NUMBER.sub("", line)  # drop the leading "C36 " card marker
        for match in _BYTE_HINT.finditer(line):
            label = " ".join(match.group(1).split()).strip(" ,;")
            found.append((label, int(match.group(2)), int(match.group(3))))
    return found


class SegySource:
    """A SEG-Y file opened without scanning geometry.

    Opening is cheap whatever the file size. Only :meth:`index` reads trace
    headers (one pass), and only :meth:`read_traces` reads samples, for the
    traces asked for.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        header_map: HeaderMap | None = None,
        *,
        endian: str = "big",
        domain: SampleDomain | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        """Open ``path``. ``endian`` is ``"big"`` (standard) or ``"little"``.

        ``domain`` states the vertical domain and quantity; if omitted, a choice
        remembered for this file, then the file name and keyed text-header lines,
        are tried (see :mod:`ep_seismic.volume.domain`). ``header_map`` says which trace
        header bytes hold inline, crossline and coordinates; if omitted, a map
        remembered for this file is used, else the SEG-Y rev 1 defaults. See
        :attr:`header_map_origin`. ``cache_dir`` holds the index and the remembered
        domain and header-map sidecars.

        Raises
        ------
        ValueError
            If the file has no traces or its size does not fit fixed-length traces.
        """
        self.path = Path(path)
        self.endian = endian
        self._cache_dir = cache_dir
        self._domain_arg = domain
        self.header_map, self.header_map_origin = self._resolve_header_map(header_map)
        try:
            self._f = segyio.open(
                self.path, "r", endian=endian, strict=False, ignore_geometry=True
            )
        except RuntimeError as exc:
            raise ValueError(f"cannot open {self.path} as SEG-Y: {exc}") from exc
        self.n_traces: int = self._f.tracecount
        self.n_samples: int = len(self._f.samples)
        if self.n_traces == 0:
            self._f.close()
            raise ValueError(f"{self.path} has no traces")
        first = self._f.header[0]
        self._samples, self.axis_notes = sample_axis(
            self.n_samples,
            float(segyio.tools.dt(self._f)) / 1000.0,
            int(first[TraceField.DelayRecordingTime]),
            int(first[TraceField.ScalarTraceHeader]),
            self.text_header,
        )

    def __enter__(self) -> SegySource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._f.close()

    @property
    def samples(self) -> np.ndarray:
        """Sample axis from the headers: ms if time, m (or ft) if depth.

        Built by :func:`sample_axis` from the first trace's delay recording time,
        its time scalar (applied only when it is a value SEG-Y allows) and the
        sample interval; :attr:`axis_notes` says when a header was disregarded.
        """
        return self._samples

    @property
    def measurement_system(self) -> int:
        """Raw SEG-Y binary header code: 1 metres, 2 feet, 0 unset."""
        return int(self._f.bin[BinField.MeasurementSystem])

    @property
    def text_header(self) -> str:
        return _decode_text_header(bytes(self._f.text[0]))

    # -- header map -----------------------------------------------------------

    def _header_sidecar(self) -> Path:
        return index_path(self.path, self._cache_dir).with_suffix(".headers.json")

    def _resolve_header_map(self, explicit: HeaderMap | None) -> tuple[HeaderMap, str]:
        """Explicit map, else one remembered for this file, else the defaults."""
        if explicit is not None:
            return explicit, "user"
        sidecar = self._header_sidecar()
        if sidecar.exists():
            try:
                remembered = HeaderMap.from_json(sidecar.read_text(encoding="utf-8"))
                return remembered, "remembered"
            except (ValueError, TypeError, KeyError, OSError):
                pass  # unreadable sidecar: fall back to the defaults
        return HeaderMap(), "default"

    def set_header_map(self, header_map: HeaderMap, *, remember: bool = True) -> None:
        """Use this header map from now on; by default remember it for this file.

        A cached index built with a different map is rebuilt on the next
        :meth:`index` call.
        """
        self.header_map, self.header_map_origin = header_map, "user"
        if remember:
            sidecar = self._header_sidecar()
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(header_map.to_json(), encoding="utf-8")

    def header_hints(self) -> list[tuple[str, int, int]]:
        """Byte positions the text header claims, as ``(label, first, last)``."""
        return header_hints(self.text_header)

    def inspect_headers(self, traces: Sequence[int] | None = None) -> dict:
        """Raw values of the mapped header fields at a few traces.

        Defaults to the first, middle and last trace. Returns ``{"traces": [...],
        "fields": {name: {"byte": b, "values": [...]}}}``. Values are the stored
        integers; coordinates are not scaled.
        """
        n = self.n_traces
        picked = sorted({0, n // 2, n - 1}) if traces is None else list(traces)
        cols = self._read_header_rows(np.array(picked), self._index_fields())
        fields = {}
        for name, (byte, _) in self._index_fields().items():
            fields[name] = {"byte": byte, "values": cols[name].tolist()}
        return {"traces": picked, "fields": fields}

    def describe_headers(self, traces: Sequence[int] | None = None) -> str:
        """A readable table of the mapped fields, the text header's byte claims, and
        which map is in use, to check before trusting an index.
        """
        info = self.inspect_headers(traces)
        lines = [
            f"Header map in use: {self.header_map_origin}",
            f"Traces inspected: {info['traces']}",
        ]
        for name, f in info["fields"].items():
            lines.append(f"  {name:<10} byte {f['byte']:>3}: {f['values']}")
        claims = self.header_hints()
        if claims:
            lines.append("Text header says (unverified):")
            lines.extend(f"  {label}: bytes {a}-{b}" for label, a, b in claims)
        return "\n".join(lines)

    # -- domain ---------------------------------------------------------------

    def _domain_sidecar(self) -> Path:
        return index_path(self.path, self._cache_dir).with_suffix(".domain.json")

    def detect(self):
        """What the file name, folder name and text header say about the domain."""
        return detect_domain(self.path, self.text_header)

    @property
    def domain(self) -> SampleDomain | None:
        """Explicit choice, else remembered choice, else detected; None if unsettled."""
        if self._domain_arg is not None:
            return self._domain_arg
        sidecar = self._domain_sidecar()
        if sidecar.exists():
            try:
                remembered = domain_from_json(sidecar.read_text(encoding="utf-8"))
                return replace(remembered, origin="remembered")
            except (ValueError, TypeError, KeyError, OSError):
                pass  # unreadable sidecar: fall through to detection
        return domain_from_detection(self.detect(), self.measurement_system)

    def check_domain(self, **kwargs):
        """A plausibility check of time vs depth on a few hundred traces; it never
        sets the domain. See :func:`ep_seismic.volume.domain_check.check_domain`."""
        from ep_seismic.volume.domain_check import check_domain

        return check_domain(self, **kwargs)

    def set_domain(self, domain: SampleDomain, *, remember: bool = True) -> None:
        """Set the domain for this source; by default remember it for this file."""
        self._domain_arg = domain
        if remember:
            sidecar = self._domain_sidecar()
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(domain_to_json(domain), encoding="utf-8")

    def require_domain(self) -> SampleDomain:
        """The domain, or raise :class:`DomainUnspecified` saying what to provide."""
        found = self.domain
        if found is None:
            raise DomainUnspecified(unspecified_message(self.detect(), self.path))
        return found

    # -- reading traces -------------------------------------------------------

    def read_traces(
        self,
        indices: Sequence[int] | np.ndarray,
        *,
        sample_slice: slice | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> np.ndarray:
        """Read the given traces, in the given order, as ``float32``.

        ``sample_slice`` keeps only part of each trace (a time or depth window);
        traces are contiguous on disk, so this trims memory, not disk reads.

        Disk access is kept sequential: traces are read in file order, runs of
        adjacent traces in a single call, then returned in the order asked for
        (repeats allowed). Peak extra memory is one run of at most
        ``_RUN_BYTES`` bytes.

        Raises
        ------
        ValueError
            If an index is out of range, or the result would exceed ``max_bytes``.
        """
        idx = np.asarray(indices)
        if idx.size == 0 and idx.ndim == 1:
            idx = idx.astype(np.intp)  # an empty list arrives as float64
        if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError("indices must be a 1-D sequence of integers")
        if idx.size and (idx.min() < 0 or idx.max() >= self.n_traces):
            raise ValueError(f"trace index outside 0..{self.n_traces - 1}")
        window = sample_slice if sample_slice is not None else slice(None)
        n_out = len(range(*window.indices(self.n_samples)))
        nbytes = idx.size * n_out * 4
        if nbytes > max_bytes:
            raise ValueError(
                f"{idx.size} traces x {n_out} samples is "
                f"{nbytes / 1e9:.2f} GB, over the {max_bytes / 1e9:.2f} GB cap; "
                "extract fewer traces or raise max_bytes"
            )
        out = np.empty((idx.size, n_out), dtype=np.float32)
        order = np.argsort(idx, kind="stable")
        sorted_idx = idx[order]
        max_run = max(1, _RUN_BYTES // (self.n_samples * 4))
        for start, stop in contiguous_runs(sorted_idx, max_run):
            first = int(sorted_idx[start])
            rows = self._read_run(first, first + (stop - start))
            out[order[start:stop]] = rows[:, window]
        return out

    def _read_run(self, first: int, stop: int) -> np.ndarray:
        """Traces ``first`` to ``stop - 1`` in one read, as ``float32``."""
        return self._f.trace.raw[first:stop]

    # -- trace headers --------------------------------------------------------

    def _layout(self) -> tuple[int, int]:
        """File offset of trace 0 and bytes per trace, checked against file size."""
        width = _SAMPLE_BYTES.get(int(self._f.format))
        if width is None:
            raise ValueError(f"unsupported sample format {self._f.format}")
        trace_bytes = _TRACE_HEADER_BYTES + self.n_samples * width
        first = self.path.stat().st_size - self.n_traces * trace_bytes
        if first < _FILE_HEADER_BYTES or (first - _FILE_HEADER_BYTES) % 3200:
            raise ValueError(
                f"{self.path} size does not fit {self.n_traces} fixed-length traces"
                f" of {trace_bytes} bytes; the file may be truncated or have"
                " variable-length traces"
            )
        return first, trace_bytes

    def _parse(
        self, head: np.ndarray, fields: dict[str, tuple[int, int]]
    ) -> dict[str, np.ndarray]:
        """Pull named integer fields out of an array of raw trace headers."""
        order = ">" if self.endian == "big" else "<"
        out = {}
        for name, (byte, w) in fields.items():
            col = np.ascontiguousarray(head[:, byte - 1 : byte - 1 + w])
            out[name] = col.view(f"{order}i{w}").ravel().astype(f"i{w}")
        return out

    def _iter_header_chunks(
        self, fields: dict[str, tuple[int, int]], chunk: int = 200_000
    ) -> Iterator[tuple[int, int, dict[str, np.ndarray]]]:
        """Yield ``(first, last, fields)`` for consecutive chunks of traces.

        One pass: every trace header is touched once, however many fields are
        read. Field-by-field reads would page through a multi-TB file once per
        field. Memory is bounded by ``chunk``.
        """
        first, trace_bytes = self._layout()
        raw = np.memmap(
            self.path,
            dtype=np.uint8,
            mode="r",
            offset=first,
            shape=(self.n_traces, trace_bytes),
        )
        try:
            for a in range(0, self.n_traces, chunk):
                b = min(a + chunk, self.n_traces)
                head = np.ascontiguousarray(raw[a:b, :_TRACE_HEADER_BYTES])
                yield a, b, self._parse(head, fields)
        finally:
            del raw  # release the file handle (Windows keeps it locked otherwise)

    def _scan_headers(
        self, fields: dict[str, tuple[int, int]], chunk: int = 200_000
    ) -> dict[str, np.ndarray]:
        """All traces' values of ``fields`` as full arrays (one pass)."""
        out = {
            k: np.empty(self.n_traces, dtype=f"i{w}") for k, (_, w) in fields.items()
        }
        for a, b, cols in self._iter_header_chunks(fields, chunk):
            for name, values in cols.items():
                out[name][a:b] = values
        return out

    def _read_header_rows(
        self, indices: np.ndarray, fields: dict[str, tuple[int, int]]
    ) -> dict[str, np.ndarray]:
        """Header fields of just the listed traces (random access)."""
        first, trace_bytes = self._layout()
        raw = np.memmap(
            self.path,
            dtype=np.uint8,
            mode="r",
            offset=first,
            shape=(self.n_traces, trace_bytes),
        )
        try:
            head = np.ascontiguousarray(raw[np.asarray(indices), :_TRACE_HEADER_BYTES])
        finally:
            del raw
        return self._parse(head, fields)

    def _index_fields(self) -> dict[str, tuple[int, int]]:
        hm = self.header_map
        return {
            "inline": (hm.inline, 4),
            "crossline": (hm.crossline, 4),
            "offset": (hm.offset, 4),
            "scalar": (hm.scalar, 2),
            "cdp_x": (hm.cdp_x, 4),
            "cdp_y": (hm.cdp_y, 4),
            "source_x": (hm.source_x, 4),
            "source_y": (hm.source_y, 4),
        }

    def _make_meta(self, xy_fields: str, warnings: list[str]) -> IndexMeta:
        dt = np.diff(self.samples[:2]) if self.n_samples > 1 else np.array([0.0])
        if dt[0] <= 0:
            warnings.append("sample interval is zero or negative")
        st = self.path.stat()
        return IndexMeta(
            source_path=str(self.path.resolve()),
            source_size=st.st_size,
            source_mtime_ns=st.st_mtime_ns,
            n_traces=self.n_traces,
            n_samples=self.n_samples,
            sample_start=float(self.samples[0]),
            sample_step=float(dt[0]),
            xy_fields=xy_fields,
            measurement_system=self.measurement_system,
            coordinate_units=int(self._f.header[0][TraceField.CoordinateUnits]),
            warnings=(*warnings, *self.axis_notes),
            header_map=self.header_map.to_json(),
        )

    # -- indexes --------------------------------------------------------------

    def build_index(self, extra_warnings: Sequence[str] = ()) -> TraceIndex:
        """Per-trace index: read the trace headers once (not saved)."""
        warnings: list[str] = list(extra_warnings)
        cols = self._scan_headers(self._index_fields())
        inline, crossline = cols["inline"], cols["crossline"]
        offset, scalar = cols["offset"], cols["scalar"]
        if not inline.any() or not crossline.any():
            warnings.append(
                "inline or crossline header is all zero; line selection will not work"
            )
        if inline.min() < 0 or crossline.min() < 0:
            warnings.append(
                "negative inline or crossline numbers: the trace headers may not "
                "follow the SEG-Y rev 1 layout; check the text header and HeaderMap"
            )

        cx, cy = cols["cdp_x"], cols["cdp_y"]
        xy_fields = "cdp_x/cdp_y"
        if not cx.any() and not cy.any():
            cx, cy = cols["source_x"], cols["source_y"]
            xy_fields = "source_x/source_y"
            warnings.append("CDP X/Y all zero; used source X/Y instead")
            if not cx.any() and not cy.any():
                warnings.append("source X/Y also all zero; no usable coordinates")
        warnings.extend(_scalar_warnings(int(scalar.min()), int(scalar.max())))
        return TraceIndex(
            inline=inline.astype(np.int32),
            crossline=crossline.astype(np.int32),
            offset=offset.astype(np.int32),
            x=apply_scalar(cx, scalar),
            y=apply_scalar(cy, scalar),
            meta=self._make_meta(xy_fields, warnings),
        )

    def build_lattice(
        self, *, xy_tolerance: float = 1.0, chunk: int = 200_000
    ) -> tuple[LatticeIndex | None, str]:
        """Try to describe the whole file as a regular grid, checking every trace.

        Returns ``(index, "")`` on success, or ``(None, reason)`` if the file is
        not a complete, ordered inline x crossline grid with one trace per cell
        and affine coordinates (to within ``xy_tolerance`` coordinate units).
        Memory is bounded by ``chunk`` however large the file is.
        """
        n = self.n_traces
        if n < 4:
            return None, "too few traces for a grid"
        fields = self._index_fields()

        def rows(idx: list[int]) -> dict[str, np.ndarray]:
            return self._read_header_rows(np.array(idx), fields)

        r = rows([0, 1])
        il0, xl0 = int(r["inline"][0]), int(r["crossline"][0])
        if r["inline"][1] == il0 and r["crossline"][1] != xl0:
            order, slow_key, fast_key = "inline_major", "inline", "crossline"
        elif r["crossline"][1] == xl0 and r["inline"][1] != il0:
            order, slow_key, fast_key = "xline_major", "crossline", "inline"
        else:
            return None, "first two traces do not step along a single line"
        slow0 = il0 if order == "inline_major" else xl0
        fast_step = int(r[fast_key][1] - r[fast_key][0])

        def slow_at(k: int) -> int:
            return int(rows([k])[slow_key][0])

        lo, hi = 1, 2  # invariant: slow_at(lo) == slow0; slow_at(hi) != slow0
        while hi < n - 1 and slow_at(hi) == slow0:
            lo, hi = hi, hi * 2
        hi = min(hi, n - 1)
        if slow_at(hi) == slow0:
            return None, "a single line, not a grid"
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if slow_at(mid) == slow0:
                lo = mid
            else:
                hi = mid
        n_fast = hi
        if n % n_fast:
            return (
                None,
                f"trace count {n} is not a whole number of {n_fast}-trace lines",
            )
        n_slow = n // n_fast
        if n_fast < 2 or n_slow < 2:
            return None, "fewer than 2 lines in one direction"
        slow_step = slow_at(n_fast) - slow0

        ends = [0, n_fast - 1, (n_slow - 1) * n_fast, n - 1]
        e = rows(ends)
        use_cdp = bool(e["cdp_x"].any() or e["cdp_y"].any())
        xk, yk = ("cdp_x", "cdp_y") if use_cdp else ("source_x", "source_y")
        xy_fields = "cdp_x/cdp_y" if use_cdp else "source_x/source_y"
        px = apply_scalar(e[xk], e["scalar"])
        py = apply_scalar(e[yk], e["scalar"])
        p0 = np.array([px[0], py[0]])
        d_fast = (np.array([px[1], py[1]]) - p0) / (n_fast - 1)
        d_slow = (np.array([px[2], py[2]]) - p0) / (n_slow - 1)
        if abs(d_fast[0] * d_slow[1] - d_fast[1] * d_slow[0]) < 1e-9:
            return None, "line directions are not independent in X/Y"
        offset0 = int(e["offset"][0])

        max_res, smin, smax = 0.0, 32767, -32768
        for a, b, c in self._iter_header_chunks(fields, chunk):
            k = np.arange(a, b, dtype=np.int64)
            si, fj = np.divmod(k, n_fast)
            pred_slow = slow0 + slow_step * si
            pred_fast = (xl0 if order == "inline_major" else il0) + fast_step * fj
            got_slow, got_fast = c[slow_key], c[fast_key]
            bad = (
                (got_slow != pred_slow)
                | (got_fast != pred_fast)
                | (c["offset"] != offset0)
            )
            if bad.any():
                first_bad = a + int(np.flatnonzero(bad)[0])
                return None, f"trace {first_bad} breaks the regular line/offset pattern"
            x = apply_scalar(c[xk], c["scalar"])
            y = apply_scalar(c[yk], c["scalar"])
            res = np.hypot(
                x - (p0[0] + si * d_slow[0] + fj * d_fast[0]),
                y - (p0[1] + si * d_slow[1] + fj * d_fast[1]),
            )
            max_res = max(max_res, float(res.max()))
            if max_res > xy_tolerance:
                first_bad = a + int(np.argmax(res > xy_tolerance))
                return None, (
                    f"coordinates at trace {first_bad} are {max_res:.2f} from a "
                    f"regular grid (tolerance {xy_tolerance})"
                )
            smin, smax = (
                min(smin, int(c["scalar"].min())),
                max(smax, int(c["scalar"].max())),
            )

        warnings = _scalar_warnings(smin, smax)
        if not use_cdp:
            warnings.append("CDP X/Y all zero; used source X/Y instead")
        if order == "inline_major":
            il_step, xl_step, d_il, d_xl = slow_step, fast_step, d_slow, d_fast
            n_il, n_xl = n_slow, n_fast
        else:
            il_step, xl_step, d_il, d_xl = fast_step, slow_step, d_fast, d_slow
            n_il, n_xl = n_fast, n_slow
        return (
            LatticeIndex(
                meta=self._make_meta(xy_fields, warnings),
                order=order,
                n_il=n_il,
                n_xl=n_xl,
                il0=il0,
                il_step=il_step,
                xl0=xl0,
                xl_step=xl_step,
                p0=(float(p0[0]), float(p0[1])),
                d_il=(float(d_il[0]), float(d_il[1])),
                d_xl=(float(d_xl[0]), float(d_xl[1])),
                offset=offset0,
                max_xy_residual=max_res,
            ),
            "",
        )

    def index(
        self,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        rebuild: bool = False,
        compact: bool = True,
    ) -> GeometryIndex:
        """Load the sidecar index if it is current, else build and save it.

        With ``compact`` (default) a regular 3D grid is stored as a tiny
        :class:`LatticeIndex`; anything else, or ``compact=False``, gets the
        per-trace :class:`TraceIndex`, with the reason in ``meta.warnings``.
        """
        sidecar = index_path(self.path, cache_dir or self._cache_dir)
        if sidecar.exists() and not rebuild:
            try:
                cached = load_index(sidecar)
            except (ValueError, KeyError, OSError):
                cached = None
            wants_full = not compact and isinstance(cached, LatticeIndex)
            if (
                cached is not None
                and not wants_full
                and cached.is_current_for(self.path)
                and cached.meta.header_map == self.header_map.to_json()
                and cached.meta.sample_start == float(self.samples[0])
            ):  # an index built with an older sample axis is rebuilt, not reused
                return cached
        built: GeometryIndex | None = None
        reason = ""
        if compact:
            built, reason = self.build_lattice()
        if built is None:
            extra = (
                [f"not a regular grid ({reason}); per-trace index used"]
                if reason
                else []
            )
            built = self.build_index(extra)
        built.save(sidecar)
        return built


def _scalar_warnings(smin: int, smax: int) -> list[str]:
    out = []
    if smin != smax:
        out.append("coordinate scalar varies between traces; applied per trace")
    if smin < -10000 or smax > 10000:
        out.append("coordinate scalar outside +-10000; check the headers")
    return out


def open_segy(
    path: str | os.PathLike[str],
    header_map: HeaderMap | None = None,
    *,
    endian: str = "big",
    domain: SampleDomain | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> SegySource:
    """Open a SEG-Y file in place."""
    return SegySource(
        path, header_map, endian=endian, domain=domain, cache_dir=cache_dir
    )

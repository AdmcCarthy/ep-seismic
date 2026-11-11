"""SEG-Y reader for post-stack volumes (no extra dependency: numpy only).

Reads the textual header (EBCDIC), the binary header, the trace headers (inline, crossline, X, Y by byte position) and traces of a fixed-length, big-endian SEG-Y file with IBM or IEEE float samples.  Byte positions of the trace
header words are DECLARED by the caller (adapter) and checked against the file's own text header by the processing step; nothing is guessed here.  Only this module and the pipeline touch the file.
"""
from __future__ import annotations

import os
import re
import struct
from typing import Any

import numpy as np
import pandas as pd

TEXT_BYTES, BIN_BYTES, TRACE_HEAD = 3200, 400, 240
SAMPLE_BYTES = {1: 4, 2: 4, 5: 4, 8: 1}       # IBM float, int32, IEEE float, int8 (the formats a stack can use)


def read_text_header(path: str) -> list[str]:
    """The 40 card images of the textual header; EBCDIC when the file is EBCDIC, else ASCII (decided by which decoding gives more letters)."""
    with open(path, "rb") as f:
        raw = f.read(TEXT_BYTES)
    eb, asc = raw.decode("cp037", errors="replace"), raw.decode("latin-1", errors="replace")
    letters = lambda s: sum(c.isascii() and c.isalpha() for c in s)       # accented letters of a wrong Latin-1 reading of EBCDIC must not count
    text = eb if letters(eb) >= letters(asc) else asc
    return [text[i:i + 80].rstrip() for i in range(0, TEXT_BYTES, 80)]


def read_binary_header(path: str) -> dict[str, Any]:
    """The binary header words of the standard (revision 0/1): sample interval [us], samples per trace, sample format code, measurement system, revision, fixed-length flag and trace polarity (0 = not stated)."""
    with open(path, "rb") as f:
        f.seek(TEXT_BYTES)
        b = f.read(BIN_BYTES)

    def h(pos: int) -> int:                                       # 1-based byte position in the file, 2-byte big-endian
        return struct.unpack(">h", b[pos - TEXT_BYTES - 1:pos - TEXT_BYTES + 1])[0]

    size = os.path.getsize(path)
    ns, fmt = h(3221), h(3225)
    tl = TRACE_HEAD + ns * SAMPLE_BYTES.get(fmt, 4)
    ext = h(3505)                                                 # number of extended textual headers (0 when none)
    return {"interval_us": h(3217), "samples": ns, "format_code": fmt, "measurement_system": h(3255), "polarity_code": h(3257), "revision": h(3501), "fixed_length": h(3503), "extended_headers": ext,
            "trace_bytes": tl, "file_bytes": size, "traces": int((size - TEXT_BYTES - BIN_BYTES - ext * TEXT_BYTES) // tl)}


def _traces_view(path: str, bh: dict[str, Any]) -> np.memmap:
    if bh["extended_headers"]:
        raise ValueError("extended textual headers are not supported: the offset of the first trace would be a guess")
    return np.memmap(path, dtype=np.uint8, mode="r", offset=TEXT_BYTES + BIN_BYTES, shape=(bh["traces"], bh["trace_bytes"]))


def _i4(mm: np.memmap, pos: int) -> np.ndarray:
    """A 4-byte big-endian integer trace header word at 1-based byte `pos` of the trace header, for every trace."""
    return np.ascontiguousarray(mm[:, pos - 1:pos + 3]).view(">i4").ravel().astype(np.int64)


def _i2(mm: np.memmap, pos: int) -> np.ndarray:
    return np.ascontiguousarray(mm[:, pos - 1:pos + 1]).view(">i2").ravel().astype(np.int64)


def read_trace_headers(path: str, bh: dict[str, Any], words: dict[str, int]) -> pd.DataFrame:
    """Per trace: inline, crossline, X, Y (integers as stored), the coordinate scalar (bytes 71-72), the recording delay [ms] (bytes 109-110) and the number of samples in the trace header (bytes 115-116).
    `words` gives the byte positions: inline, crossline, x, y."""
    mm = _traces_view(path, bh)
    return pd.DataFrame({"trace": np.arange(bh["traces"]), "inline": _i4(mm, words["inline"]), "crossline": _i4(mm, words["crossline"]), "x_raw": _i4(mm, words["x"]), "y_raw": _i4(mm, words["y"]),
                         "scalar": _i2(mm, 71), "delay_ms": _i2(mm, 109), "samples_in_header": _i2(mm, 115)})


def ibm_to_float(a: np.ndarray) -> np.ndarray:
    """IBM System/360 single precision (big-endian uint32) to float32: sign, 7-bit base-16 exponent (bias 64), 24-bit fraction."""
    a = a.astype(np.uint32)
    sign = np.where(a >> 31, -1.0, 1.0)
    exp = ((a >> 24) & 0x7F).astype(np.int32) - 64
    frac = (a & 0x00FFFFFF).astype(np.float64) / float(1 << 24)
    return (sign * frac * np.power(16.0, exp)).astype(np.float32)


def read_traces(path: str, bh: dict[str, Any], index: np.ndarray) -> np.ndarray:
    """The samples of traces `index` (0-based trace numbers) as float32, shape (n, samples); rows follow `index`."""
    mm = _traces_view(path, bh)
    ns = bh["samples"]
    order = np.argsort(index, kind="stable")
    raw = np.ascontiguousarray(mm[np.asarray(index)[order], TRACE_HEAD:TRACE_HEAD + 4 * ns]).view(">u4" if bh["format_code"] in (1, 2) else ">f4")
    if bh["format_code"] == 1:
        out = ibm_to_float(raw.astype(np.uint32))
    elif bh["format_code"] == 2:
        out = raw.astype(np.int32).astype(np.float32)
    elif bh["format_code"] == 5:
        out = raw.astype(np.float32)
    else:
        raise ValueError(f"sample format code {bh['format_code']} is not supported")
    res = np.empty_like(out)
    res[order] = out
    return res


def text_header_facts(lines: list[str]) -> dict[str, Any]:
    """Facts the survey's own text header states (read by pattern, as stated, None when absent): processing grid corner points, bin size, azimuth, the byte positions, the coordinate multiplier, first / last sample and interval."""
    t = "\n".join(lines)
    out: dict[str, Any] = {}
    corners = {}
    for m in re.finditer(r"PG(\d)\s+X:\s*([\d.]+)\s+Y:\s*([\d.]+)\s+IL:\s*(\d+)\s+XL:\s*(\d+)", t):
        corners[int(m.group(1))] = {"x": float(m.group(2)), "y": float(m.group(3)), "inline": int(m.group(4)), "crossline": int(m.group(5))}
    out["corners"] = corners
    for key, pat in {"bin_inline_m": r"INLINE\s+BIN SIZE:\s*([\d.]+)", "bin_crossline_m": r"CROSSLINE\s+BIN SIZE:\s*([\d.]+)", "azimuth_deg": r"AZIMUTH:\s*([\d.]+)", "coordinate_multiplier": r"MULTIPLIED BY\s*(\d+)",
                     "first_sample_ms": r"FIRST SAMPLE:\s*([\d.]+)", "last_sample_ms": r"LAST SAMPLE:\s*([\d.]+)", "interval_ms": r"SAMPLE INTERVAL:\s*([\d.]+)"}.items():
        m = re.search(pat, t)
        out[key] = float(m.group(1)) if m else None
    for key, pat in {"inline_byte": r"INLINE\s+BYTES\s+(\d+)-", "crossline_byte": r"CROSS LINE\s+BYTES\s+(\d+)-", "x_byte": r"UTM-X\s+BYTES\s+(\d+)-", "y_byte": r"UTM-Y\s+BYTES\s+(\d+)-"}.items():
        m = re.search(pat, t)
        out[key] = int(m.group(1)) if m else None
    m = re.search(r"DATA TYPE:\s*(.+?)\s{2,}", t)
    out["data_type"] = m.group(1).strip() if m else None
    m = re.search(r"DATE:\s*([\d-]+)", t)
    out["processing_date"] = m.group(1) if m else None
    return out

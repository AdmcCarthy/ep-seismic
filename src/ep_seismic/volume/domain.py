"""What the samples are: the vertical domain and the quantity they hold.

SEG-Y headers say neither whether the vertical axis is time or depth, nor whether
the values are amplitudes, velocities or something else. Guessing wrong makes a
section silently misleading, so this library never guesses: the domain is taken
from (in order) the user, a remembered earlier choice, or unambiguous statements
in the file's own name or a keyed line of its textual header (``VERTICAL DOMAIN:
TWT``). Folder names and bare words in the header are only hints: real files show
they mislead. If nothing settles BOTH the vertical domain and the quantity,
:class:`DomainUnspecified` is raised and the user must say.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

VERTICAL = ("time", "depth")
QUANTITIES = ("amplitude", "velocity", "impedance", "density", "other")
_UNITS_BY_SYSTEM = {1: "m", 2: "ft"}  # SEG-Y binary header measurement system


class DomainUnspecified(ValueError):
    """The vertical domain or the sample quantity is not known."""


@dataclass(frozen=True)
class SampleDomain:
    """Vertical domain and quantity of a seismic volume.

    ``vertical`` is ``"time"`` (unit ms, as SEG-Y states the sample interval in
    microseconds) or ``"depth"`` (unit ``"m"`` or ``"ft"``). ``quantity`` is one of
    :data:`QUANTITIES`; for ``"other"`` give ``quantity_name``. ``reference`` is an
    optional depth or time datum such as ``"TVDSS"`` or ``"MSL"``. ``origin`` and
    ``evidence`` say how it was decided and are shown with every section.
    """

    vertical: str
    vertical_unit: str
    quantity: str
    quantity_unit: str
    origin: str
    evidence: str = ""
    quantity_name: str = ""
    reference: str = ""

    def __post_init__(self) -> None:
        if self.vertical not in VERTICAL:
            raise ValueError(
                f"vertical must be one of {VERTICAL}, got {self.vertical!r}"
            )
        ok_units = ("ms",) if self.vertical == "time" else ("m", "ft")
        if self.vertical_unit not in ok_units:
            raise ValueError(f"{self.vertical} unit must be one of {ok_units}")
        if self.quantity not in QUANTITIES:
            raise ValueError(f"quantity must be one of {QUANTITIES}")
        if self.quantity == "other" and not self.quantity_name:
            raise ValueError("quantity 'other' needs quantity_name")

    @property
    def vertical_label(self) -> str:
        name = "TWT" if self.vertical == "time" else "Depth"
        ref = f" {self.reference}" if self.reference else ""
        return f"{name}{ref} [{self.vertical_unit}]"

    @property
    def quantity_label(self) -> str:
        name = self.quantity_name if self.quantity == "other" else self.quantity
        return f"{name.capitalize()} [{self.quantity_unit}]"

    def describe(self) -> str:
        how = f"{self.origin}: {self.evidence}" if self.evidence else self.origin
        return f"{self.vertical_label}; {self.quantity_label} ({how})"


def make_domain(
    vertical: str,
    quantity: str,
    *,
    vertical_unit: str | None = None,
    quantity_unit: str | None = None,
    quantity_name: str = "",
    reference: str = "",
    measurement_system: int = 0,
    origin: str = "user",
    evidence: str = "",
) -> SampleDomain:
    """Build a :class:`SampleDomain`, filling units only where the file states them.

    Time is always ms. A depth or velocity unit comes from ``vertical_unit`` /
    ``quantity_unit`` or, failing that, the SEG-Y measurement-system code
    (1 metres, 2 feet). Amplitude is unitless unless told otherwise.

    Raises
    ------
    DomainUnspecified
        If a needed unit is neither given nor stated in the file.
    """
    sys_unit = _UNITS_BY_SYSTEM.get(measurement_system)
    if vertical_unit is None:
        if vertical == "time":
            vertical_unit = "ms"
        elif sys_unit is not None:
            vertical_unit = sys_unit
        else:
            raise DomainUnspecified(
                "depth unit not stated in the file: pass vertical_unit='m' or 'ft'"
            )
    if quantity_unit is None:
        if quantity == "amplitude":
            quantity_unit = "unitless"
        elif quantity == "velocity" and sys_unit is not None:
            quantity_unit = f"{sys_unit}/s"
        elif quantity == "velocity":
            raise DomainUnspecified(
                "velocity unit not stated in the file: "
                "pass quantity_unit='m/s' or 'ft/s'"
            )
        else:
            raise DomainUnspecified(f"pass quantity_unit for quantity {quantity!r}")
    return SampleDomain(
        vertical=vertical,
        vertical_unit=vertical_unit,
        quantity=quantity,
        quantity_unit=quantity_unit,
        origin=origin,
        evidence=evidence,
        quantity_name=quantity_name,
        reference=reference,
    )


# Whole-word tokens. File and folder names may use plain words; the free-text
# header may not ("TIME" turns up in dates), so it gets a stricter list.
_NAME_TOKENS = {
    "vertical": {
        "time": {"twt", "time", "pstm", "owt"},
        "depth": {"depth", "tvd", "tvdss"},
    },
    "quantity": {
        "amplitude": {"amp", "amplitude", "stack", "stk", "migrated"},
        "velocity": {"vel", "velocity", "vint", "vrms", "vavg"},
        "impedance": {"impedance"},
        "density": {"density", "rhob", "rho"},
    },
}
# The text header decides only when it STATES the answer in a keyed line such as
# "VERTICAL DOMAIN: TWT" or "DATA TYPE: AMPLITUDE". A bare mention proves nothing:
# a real Volve VSP header says "TVDSS: BYTE 53-56" (a trace header label) for a
# file whose samples are time. Bare mentions and folder names are only hints.
# "PSDM" names the migration (pre-stack DEPTH migration), not the axis of the output:
# the Volve stack ST10010ZC11_PZ_PSDM_KIRCH_FULL_T is PSDM depth-converted back to TIME
# (its "_T" and "DEPTH TO TIME CONVERSION" say so), so the word is a hint only.
_NAME_HINT_TOKENS = {"psdm": "a depth-migration name; its output may still be in time"}
_KEYED = {
    "vertical": re.compile(
        r"\b(?:vertical|sample|z)[\s_-]*(?:domain|axis|type)\s*[:=]\s*([A-Za-z0-9]+)",
        re.IGNORECASE,
    ),
    "quantity": re.compile(
        r"\b(?:data|volume|attribute|quantity)[\s_-]*(?:type|quantity|attribute)?"
        r"\s*[:=]\s*([A-Za-z0-9]+)",
        re.IGNORECASE,
    ),
}


def _words(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", text.lower()) if w}


@dataclass(frozen=True)
class Detection:
    """What the file itself STATES about its domain, with the evidence for each.

    ``vertical`` and ``quantity`` are set only by decisive evidence (the file's own
    name, or a keyed line in the text header). ``hints`` lists weaker clues
    (folder name, bare mentions in the text header) that never settle anything.
    """

    vertical: str | None = None
    quantity: str | None = None
    vertical_evidence: str = ""
    quantity_evidence: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)
    hints: tuple[str, ...] = field(default_factory=tuple)

    @property
    def complete(self) -> bool:
        return self.vertical is not None and self.quantity is not None


def _hits(words: set[str], tokens: dict[str, dict[str, set[str]]], key: str):
    return {
        value: sorted(words & toks)
        for value, toks in tokens[key].items()
        if words & toks
    }


def detect_domain(
    path: str | Path,
    text_header: str = "",
) -> Detection:
    """Look for what the file states about its domain.

    Decisive: whole words in the file's own name, and keyed lines in the text
    header (``VERTICAL DOMAIN: TWT``, ``DATA TYPE: AMPLITUDE``). A value is
    reported only if the decisive evidence points to exactly one answer; any
    conflict leaves it unset and is explained in ``notes``. Folder-name words and
    bare words in the text header are collected as ``hints`` only.
    """
    p = Path(path)
    found_by: dict[str, dict[str, list[str]]] = {"vertical": {}, "quantity": {}}
    hints: list[str] = []
    name_words = _words(p.stem)
    for token, why in _NAME_HINT_TOKENS.items():
        if token in name_words:
            hints.append(f"file name contains {token.upper()}: {why}")
    for key in ("vertical", "quantity"):
        for value, toks in _hits(name_words, _NAME_TOKENS, key).items():
            found_by[key].setdefault(value, []).append(f"file name {toks}")
        for value, toks in _hits(_words(p.parent.name), _NAME_TOKENS, key).items():
            hints.append(f"folder name suggests {key} = {value} {toks}")
        for value, toks in _hits(_words(text_header), _NAME_TOKENS, key).items():
            hints.append(f"text header mentions {key} = {value} {toks}")
        match = _KEYED[key].search(text_header)
        if match:
            word = match.group(1)
            keyed = _hits({word.lower()}, _NAME_TOKENS, key)
            if keyed:
                (value,) = keyed
                found_by[key].setdefault(value, []).append(
                    f"text header line {match.group(0).strip()!r}"
                )
    out: dict[str, tuple[str | None, str]] = {}
    notes: list[str] = []
    for key in ("vertical", "quantity"):
        found = found_by[key]
        if len(found) == 1:
            ((value, where),) = found.items()
            out[key] = (value, "; ".join(where))
        else:
            out[key] = (None, "")
            if len(found) > 1:
                notes.append(f"conflicting {key} evidence: {found}")
    return Detection(
        vertical=out["vertical"][0],
        quantity=out["quantity"][0],
        vertical_evidence=out["vertical"][1],
        quantity_evidence=out["quantity"][1],
        notes=tuple(notes),
        hints=tuple(hints),
    )


def domain_from_detection(
    det: Detection, measurement_system: int = 0
) -> SampleDomain | None:
    """A domain from detection, or None if either part (or a unit) is unsettled."""
    if not det.complete:
        return None
    try:
        return make_domain(
            det.vertical,
            det.quantity,
            measurement_system=measurement_system,
            origin="detected",
            evidence=f"vertical from {det.vertical_evidence}; "
            f"quantity from {det.quantity_evidence}",
        )
    except DomainUnspecified:
        return None


def unspecified_message(det: Detection, path: str | Path) -> str:
    """Tell the user exactly what is missing and how to supply it."""
    found = []
    if det.vertical:
        found.append(f"vertical domain = {det.vertical} ({det.vertical_evidence})")
    if det.quantity:
        found.append(f"quantity = {det.quantity} ({det.quantity_evidence})")
    parts = [
        f"The vertical domain and sample quantity of {Path(path).name} are not "
        "stated by the file name or a keyed text-header line."
    ]
    if found:
        parts.append("Found: " + "; ".join(found) + ".")
    parts.extend(det.notes)
    if det.hints:
        parts.append(
            "Hints only, NOT decisive (check them yourself): "
            + "; ".join(det.hints)
            + "."
        )
    parts.append(
        "State them, for example: src.set_domain(make_domain('time', 'amplitude')) "
        "or make_domain('depth', 'velocity', reference='TVDSS'). The choice is "
        "remembered for this file."
    )
    return " ".join(parts)


def domain_to_json(domain: SampleDomain) -> str:
    return json.dumps(asdict(domain))


def domain_from_json(text: str) -> SampleDomain:
    return SampleDomain(**json.loads(text))

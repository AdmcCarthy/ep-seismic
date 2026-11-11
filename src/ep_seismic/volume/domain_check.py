"""A plausibility check of the vertical axis: two-way time (ms) or PSDM depth (m)?

"PSDM" is a migration done in the depth domain, not a promise about the axis of the
output: such a volume is often converted back to time (the Volve stack ..._PSDM_..._T
is), so a PSDM volume can be either, and nothing here treats the word as an answer.

This NEVER sets or changes the domain: the domain stays the user's statement (see
:mod:`ep_seismic.volume.domain`). It looks for evidence that contradicts, or supports, a
reading, and says plainly when the evidence cannot tell.

**Why it often cannot tell on the Norwegian shelf.** With an average velocity of
2000 m/s one millisecond of two-way time is one metre of depth, and shelf velocities
(about 1800-2500 m/s down to the reservoir) keep the two within about 20% of each
other. A record ending at 3400 can be 3.4 s or 3.4 km; a reservoir at 2700 can be
2700 ms or 2700 m. So the magnitude of the sample values (the survey's "base") is
almost never decisive, and the water bottom is decisive only outside an overlap:

* read as time, a seabed at ``S`` units is a water depth of ``S * 1480 / 2000`` m;
* read as depth, it is ``S`` m;
* both fall inside a 60-500 m shelf profile for ``S`` between 81 and 500, so a seabed
  there says nothing. Only below 81 (depth) or above 500 (time) does it separate them.

What does separate them: the sample interval (time is 1, 2 or 4 ms; PSDM depth is
usually 5, 10, 12.5 or 25 m, and a metre value stored in the microsecond field reads as
a tiny interval), and a water depth you know. With ``water_depth`` given, the water
bottom is decisive: time puts the seabed at about 1.35 x the water depth, depth at
1.0 x.

The profiles and typical ranges below are general knowledge about the Norwegian
continental shelf, not facts about your file; change them with ``profile=`` and
``water_depth=``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ep_seismic.volume.domain import SampleDomain

# Water depth (m) that the survey is assumed to lie in, by named profile. General
# knowledge, not from the file: northern North Sea shelf about 70-300 m, the Norwegian
# Sea and Barents Sea shelf to about 500 m, the slope and deep water beyond.
WATER_DEPTH_PROFILES: dict[str, tuple[float, float]] = {
    "north_sea": (60.0, 350.0),
    "ncs_shelf": (60.0, 500.0),
    "ncs_deep": (500.0, 2500.0),
}
WATER_VELOCITY = 1480.0  # m/s, a typical North Sea value
M_PER_FT = 0.3048
# Typical record ends on the shelf (general knowledge). Time: 1-10 s. Depth imaging
# rarely goes much beyond 8 km.
TYPICAL_BASE = {"time": (1000.0, 10000.0), "depth": (1000.0, 8000.0)}

_STRONG, _WEAK, _NONE = "strong", "weak", "none"


@dataclass(frozen=True)
class Evidence:
    """One rule's finding. ``favours`` is ``"time"``, ``"depth"`` or ``None``."""

    rule: str
    favours: str | None
    strength: str
    detail: str


@dataclass(frozen=True)
class Seabed:
    """Where the first strong event is on the traces looked at (axis units)."""

    value: float
    p10: float
    p90: float
    agree: float
    n_traces: int


@dataclass(frozen=True)
class DomainCheck:
    """The outcome. ``status`` is ``"time"``, ``"depth"``, ``"leans time"``,
    ``"leans depth"``, ``"undetermined"`` or ``"conflicting"``."""

    first: float
    last: float
    step: float
    n_samples: int
    seabed: Seabed | None
    evidence: tuple[Evidence, ...]
    status: str
    profile: str
    water_depth: float | None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def report(self) -> str:
        lines = [
            f"Vertical axis: {self.first:g} to {self.last:g} in steps of {self.step:g} "
            f"({self.n_samples} samples)",
            "Water depth assumed: "
            + (
                f"{self.water_depth:g} m (given)"
                if self.water_depth is not None
                else f"profile {self.profile!r} {WATER_DEPTH_PROFILES[self.profile]} m"
            ),
        ]
        if self.seabed is not None:
            s = self.seabed
            lines.append(
                f"Seabed (first strong event): {s.value:g} "
                f"(10-90%: {s.p10:g}-{s.p90:g}; {s.agree:.0%} of {s.n_traces} "
                "traces agree)"
            )
        for e in self.evidence:
            mark = {"time": "-> time", "depth": "-> depth", None: "neutral"}[e.favours]
            lines.append(f"  [{e.strength:6s}] {e.rule}: {mark}. {e.detail}")
        lines.append(f"Verdict: {self.status.upper()}")
        lines.extend(self.notes)
        lines.append(
            "This is a plausibility check only; the domain is whatever you stated."
        )
        return "\n".join(lines)

    def against(self, domain: SampleDomain) -> str:
        """Compare with a stated domain: ``consistent``, ``inconsistent`` or ``cannot
        tell``, with the reason."""
        said = domain.vertical
        other = "depth" if said == "time" else "time"
        if self.status == said:
            return f"consistent: the evidence supports {said}"
        if self.status == f"leans {said}":
            return f"consistent (weakly): the evidence leans {said}"
        if self.status == other:
            return f"INCONSISTENT: the evidence supports {other}, you stated {said}"
        if self.status == f"leans {other}":
            return (
                f"leans against: the evidence leans {other}, you stated {said}; "
                "check it"
            )
        if self.status == "conflicting":
            return "cannot tell: the rules contradict each other; read the report"
        return (
            f"cannot tell: nothing here separates {said} from {other} "
            "(typical on the Norwegian shelf, where 1 ms is about 1 m)"
        )


def ambiguity_range(profile: str = "ncs_shelf") -> tuple[float, float]:
    """The seabed values (axis units) for which time and depth readings BOTH fall in the
    profile's water depths, so the water bottom cannot tell them apart.

    Raises
    ------
    ValueError
        For an unknown profile.
    """
    lo, hi = _profile(profile)
    k = WATER_VELOCITY / 2000.0
    return (max(lo, lo / k), min(hi, hi / k))


def _profile(name: str) -> tuple[float, float]:
    if name not in WATER_DEPTH_PROFILES:
        raise ValueError(
            f"unknown profile {name!r}; choose from {sorted(WATER_DEPTH_PROFILES)}"
        )
    return WATER_DEPTH_PROFILES[name]


def find_seabed(
    samples: np.ndarray,
    traces: np.ndarray,
    *,
    search_fraction: float = 0.4,
    level: float = 0.5,
    min_agree: float = 0.5,
    min_contrast: float = 4.0,
) -> Seabed | None:
    """The first strong event, taken as the seabed, or ``None`` if none is stable.

    Per trace: the smoothed absolute amplitude must first reach ``level`` of its
    maximum within the top ``search_fraction`` of the record, and the seabed is the
    nearest maximum after that, refined between samples by a parabola. The event must
    stand out: its smoothed amplitude at least ``min_contrast`` times the trace's
    median (white noise gives about 2, so it is not mistaken for a seabed). The traces'
    values must agree (within 25%, at least three samples of the median: a seabed
    dips over a large survey) for at least ``min_agree`` of the traces. A seabed weaker
    than something deeper in the search window, or muted out of the data, gives a wrong
    or no answer: on the real Volve stack the strongest event of the top 40% is the
    Utsira top and the shallow data are muted, so it gives none.
    """
    samples = np.asarray(samples, dtype=np.float64)
    traces = np.asarray(traces, dtype=np.float64)
    if traces.ndim != 2 or traces.shape[1] != samples.size or traces.shape[0] < 1:
        raise ValueError("traces must be (n_traces, n_samples) matching samples")
    n = traces.shape[1]
    n_search = max(8, int(search_fraction * n))
    kernel = np.ones(5) / 5.0
    env = np.apply_along_axis(
        lambda t: np.convolve(np.abs(t), kernel, mode="same"), 1, traces
    )
    found = []
    for e in env:
        top = e[:n_search]
        peak = float(top.max())
        if peak <= 0:
            continue
        first = int(np.argmax(top >= level * peak))
        if first == 0:
            continue  # energy from the first sample: no water column to see
        stop = min(n, first + 8)
        p = first + int(np.argmax(e[first:stop]))
        if e[p] < min_contrast * float(np.median(e)):
            continue
        delta = 0.0
        if 0 < p < n - 1:
            d = e[p - 1] - 2 * e[p] + e[p + 1]
            if d < 0:
                delta = float(np.clip(0.5 * (e[p - 1] - e[p + 1]) / d, -0.5, 0.5))
        found.append(samples[p] + delta * float(samples[1] - samples[0]))
    if len(found) < max(1, traces.shape[0] // 2):
        return None
    found = np.asarray(found)
    median = float(np.median(found))
    step = float(abs(samples[1] - samples[0]))
    agree = float(np.mean(np.abs(found - median) <= max(3 * step, 0.25 * abs(median))))
    if agree < min_agree:
        return None
    return Seabed(
        value=median,
        p10=float(np.percentile(found, 10)),
        p90=float(np.percentile(found, 90)),
        agree=agree,
        n_traces=int(traces.shape[0]),
    )


def data_onsets(samples: np.ndarray, traces: np.ndarray) -> np.ndarray:
    """Per trace, the axis value where it first carries energy (NaN for an all-zero
    trace): the first sample whose smoothed amplitude reaches 5% of the trace's 95th
    percentile. A trace with noise from its first sample returns the first sample."""
    samples = np.asarray(samples, dtype=np.float64)
    kernel = np.ones(5) / 5.0
    out = np.full(len(traces), np.nan)
    for i, t in enumerate(np.asarray(traces, dtype=np.float64)):
        env = np.convolve(np.abs(t), kernel, mode="same")
        ref = float(np.percentile(env, 95))
        if ref > 0:
            out[i] = samples[int(np.argmax(env >= 0.05 * ref))]
    return out


def data_onset(samples: np.ndarray, traces: np.ndarray) -> float | None:
    """Median axis value where the traces first carry energy, or ``None`` if they do not
    start blank. Processed stacks often mute the shallow data (an angle stack mutes it
    on purpose), so the water bottom may not be in the cube at all.

    Reported only if at least 10 samples and 3% of the record lie above it (see
    :func:`data_onsets`).
    """
    samples = np.asarray(samples, dtype=np.float64)
    onsets = data_onsets(samples, traces)
    onsets = onsets[np.isfinite(onsets)]
    if onsets.size == 0:
        return None
    median = float(np.median(onsets))
    k = int(round((median - samples[0]) / (samples[1] - samples[0])))
    if k < max(10, int(0.03 * samples.size)):
        return None
    return median


def interval_evidence(step: float) -> Evidence:
    """What the sample interval (ms as read from the header) says."""
    rule = "sample interval"
    if step < 0.25:
        return Evidence(
            rule,
            "depth",
            _STRONG,
            f"{step:g} would be {step * 1000:g} microseconds, not a seismic time "
            "interval; typical of a depth interval in metres stored in the "
            "microsecond field",
        )
    if step in (0.5, 1.0, 2.0, 4.0, 8.0):
        return Evidence(
            rule,
            "time",
            _WEAK,
            f"{step:g} is a common time interval; depth intervals are usually "
            "5, 10, 12.5 or 25 m",
        )
    if step in (12.5, 25.0):
        return Evidence(
            rule,
            "depth",
            _WEAK,
            f"{step:g} is a common depth interval and an unusual time interval",
        )
    if step > 32:
        return Evidence(
            rule,
            None,
            _NONE,
            f"{step:g} is too coarse for a time interval and unusual for depth; "
            "check the units",
        )
    return Evidence(rule, None, _NONE, f"{step:g} is used for both (5, 10) or neither")


def seabed_evidence(
    seabed: Seabed | None,
    *,
    profile: str,
    water_depth: float | None,
    tolerance: float,
    depth_unit: str,
) -> Evidence:
    """What the water bottom says, against a water-depth profile or a known depth."""
    rule = "seabed"
    if seabed is None:
        return Evidence(
            rule,
            None,
            _NONE,
            "no stable first strong event across the traces looked at",
        )
    to_m = M_PER_FT if depth_unit == "ft" else 1.0
    as_time = seabed.value * WATER_VELOCITY / 2000.0
    as_depth = seabed.value * to_m
    if water_depth is not None:
        err_t = abs(as_time - water_depth) / water_depth
        err_d = abs(as_depth - water_depth) / water_depth
        t_ok, d_ok = err_t <= tolerance, err_d <= tolerance
        basis, strength = f"the water depth you gave ({water_depth:g} m)", _STRONG
        lo = hi = None
        if t_ok and d_ok:  # both inside the tolerance: only a clearly closer one counts
            if err_t * 2 <= err_d:
                d_ok = False
            elif err_d * 2 <= err_t:
                t_ok = False
    else:
        lo, hi = _profile(profile)
        t_ok, d_ok = lo <= as_time <= hi, lo <= as_depth <= hi
        basis, strength = f"the {profile!r} profile ({lo:g}-{hi:g} m)", _WEAK
    said = (
        f"seabed at {seabed.value:g} is a water depth of {as_time:.0f} m read as time "
        f"(1480 m/s) or {as_depth:.0f} m read as depth, against {basis}"
    )
    if t_ok and d_ok:
        extra = ""
        if lo is not None:
            a, b = ambiguity_range(profile)
            extra = (
                f"; both fit, and for this profile the water bottom can only separate "
                f"them below {a:.0f} or above {b:.0f}"
            )
        else:
            extra = (
                f"; both fit within {tolerance:.0%}: the water depth is too close to "
                "tell (time would put the seabed at 1.35x it)"
            )
        return Evidence(rule, None, _NONE, said + extra)
    if t_ok:
        return Evidence(rule, "time", strength, said + "; only time fits")
    if d_ok:
        return Evidence(rule, "depth", strength, said + "; only depth fits")
    return Evidence(
        rule,
        None,
        _NONE,
        said + "; neither fits (a different water depth, or a wrong seabed pick)",
    )


def base_evidence(last: float, *, depth_unit: str) -> Evidence:
    """What the end of the record says (the base of the survey)."""
    rule = "base of survey"
    to_m = M_PER_FT if depth_unit == "ft" else 1.0
    t_ok = TYPICAL_BASE["time"][0] <= last <= TYPICAL_BASE["time"][1]
    d_ok = TYPICAL_BASE["depth"][0] <= last * to_m <= TYPICAL_BASE["depth"][1]
    t_lo, t_hi = TYPICAL_BASE["time"]
    d_lo, d_hi = TYPICAL_BASE["depth"]
    if t_ok and d_ok:
        return Evidence(
            rule,
            None,
            _NONE,
            f"{last:g} is a typical end for both a time record ({t_lo:g}-{t_hi:g} ms) "
            f"and a depth cube ({d_lo:g}-{d_hi:g} m): cannot separate them",
        )
    if t_ok:
        return Evidence(
            rule,
            "time",
            _WEAK,
            f"{last:g} is typical for a time record but deeper than depth cubes "
            "usually go",
        )
    if d_ok:
        return Evidence(
            rule,
            "depth",
            _WEAK,
            f"{last:g} is typical for a depth cube, not a time record",
        )
    return Evidence(rule, None, _NONE, f"{last:g} is outside the usual range for both")


def combine(evidence: Sequence[Evidence]) -> str:
    """Strong evidence rules a reading out; otherwise weak evidence is only counted."""
    possible = {"time", "depth"}
    for e in evidence:
        if e.strength == _STRONG and e.favours:
            possible &= {e.favours}
    if not possible:
        return "conflicting"
    if len(possible) == 1:
        return next(iter(possible))
    score = sum(
        (1 if e.favours == "time" else -1)
        for e in evidence
        if e.strength == _WEAK and e.favours
    )
    if score >= 1:
        return "leans time"
    if score <= -1:
        return "leans depth"
    return "undetermined"


def check_axis(
    samples: np.ndarray,
    traces: np.ndarray,
    *,
    profile: str = "ncs_shelf",
    water_depth: float | None = None,
    tolerance: float = 0.15,
    depth_unit: str = "m",
) -> DomainCheck:
    """Check a sample axis and a few traces; see :func:`check_domain` for a file.

    ``tolerance`` is how far (as a fraction) the seabed may sit from ``water_depth``
    and still fit a reading; time and depth differ by 35% there, so keep it well under
    that. When both fit, the one at least twice as close wins.

    Raises
    ------
    ValueError
        For an unknown profile or depth unit, a non-positive ``water_depth`` or
        ``tolerance``, fewer than two samples, or traces that do not match the axis.
    """
    _profile(profile)
    if depth_unit not in ("m", "ft"):
        raise ValueError("depth_unit must be 'm' or 'ft'")
    if (water_depth is not None and water_depth <= 0) or tolerance <= 0:
        raise ValueError("water_depth and tolerance must be > 0")
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size < 2:
        raise ValueError("need at least two samples")
    seabed = find_seabed(samples, traces)
    step = float(samples[1] - samples[0])
    onset = data_onset(samples, traces)
    notes = []
    if (
        seabed is not None
        and onset is not None
        and seabed.value - onset > max(0.25 * seabed.value, 10 * abs(step))
    ):
        # a real seabed is where the energy starts; a "first strong event" far below
        # the start of the data is a deeper reflector
        notes.append(
            f"The first strong event ({seabed.value:g}) lies well below where the data "
            f"start ({onset:g}): probably a deeper reflector, not the seabed, so it is "
            "not used."
        )
        seabed = None
    evidence = (
        interval_evidence(step),
        seabed_evidence(
            seabed,
            profile=profile,
            water_depth=water_depth,
            tolerance=tolerance,
            depth_unit=depth_unit,
        ),
        base_evidence(float(samples[-1]), depth_unit=depth_unit),
    )
    if seabed is None:
        if onset is not None:
            notes.append(
                f"The traces are blank above about {onset:g}: the shallow data look "
                "muted, so the water bottom may not be in the cube and the seabed rule "
                "cannot be used."
            )
    if depth_unit == "ft":
        notes.append("depth read as feet (converted to metres for the water depth)")
    return DomainCheck(
        first=float(samples[0]),
        last=float(samples[-1]),
        step=step,
        n_samples=int(samples.size),
        seabed=seabed,
        evidence=evidence,
        status=combine(evidence),
        profile=profile,
        water_depth=water_depth,
        notes=tuple(notes),
    )


def check_domain(
    src,
    *,
    n_traces: int = 300,
    profile: str = "ncs_shelf",
    water_depth: float | None = None,
    tolerance: float = 0.15,
    depth_unit: str = "m",
) -> DomainCheck:
    """Check an open source on ``n_traces`` traces spread evenly through the file.

    Reads only those traces (a few hundred, never the volume). Does not set the domain.
    """
    n = min(int(n_traces), src.n_traces)
    pick = np.unique(np.linspace(0, src.n_traces - 1, n).astype(np.int64))
    return check_axis(
        src.samples,
        src.read_traces(pick),
        profile=profile,
        water_depth=water_depth,
        tolerance=tolerance,
        depth_unit=depth_unit,
    )

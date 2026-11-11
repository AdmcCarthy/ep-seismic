import numpy as np
import pytest
import segyio

from ep_seismic.volume import (
    DomainCheck,
    ambiguity_range,
    check_axis,
    make_domain,
    open_segy,
)
from ep_seismic.volume.domain_check import data_onset, find_seabed


def ricker(n: int, centre: float, width: float) -> np.ndarray:
    """A zero-phase wavelet centred at sample ``centre`` (width in samples)."""
    a = ((np.arange(n) - centre) / width) ** 2
    return (1 - 2 * a) * np.exp(-a)


def make_traces(
    step: float, seabed: float, *, n_samples: int = 800, n: int = 60, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Samples and traces with a strong seabed at axis value ``seabed``, weaker layers
    below it and noise. Works for either axis: only the numbers matter."""
    rng = np.random.default_rng(seed)
    samples = np.arange(n_samples) * step
    traces = rng.normal(0, 0.03, (n, n_samples))
    s0 = seabed / step
    for k, (frac, amp) in enumerate(
        ((0.0, 1.0), (0.15, 0.3), (0.3, -0.25), (0.5, 0.2))
    ):
        centre = s0 + frac * (n_samples - s0)
        wobble = rng.normal(0, 0.3, (n, 1)) if k == 0 else 0
        for i in range(n):
            traces[i] += amp * ricker(
                n_samples,
                centre
                + float(np.ravel(wobble)[0] if k == 0 else 0) * 0
                + (wobble[i, 0] if k == 0 else 0),
                2.5,
            )
    return samples, traces


# -- the arithmetic behind the water-bottom test ---------------------------------


def test_the_seabed_is_decisive_only_outside_an_overlap() -> None:
    a, b = ambiguity_range("ncs_shelf")  # 60-500 m
    assert (a, b) == (pytest.approx(60 / 0.74), 500.0)  # 81 to 500
    assert ambiguity_range("north_sea")[1] == 350.0
    with pytest.raises(ValueError, match="unknown profile"):
        ambiguity_range("mars")


# -- find_seabed -----------------------------------------------------------------


def test_find_seabed_locates_a_strong_first_event() -> None:
    samples, traces = make_traces(4.0, 120.0)
    sb = find_seabed(samples, traces)
    assert sb is not None and sb.value == pytest.approx(120.0, abs=8.0)
    assert sb.agree > 0.8 and sb.n_traces == 60


def test_find_seabed_gives_none_for_noise_and_validates() -> None:
    rng = np.random.default_rng(1)
    samples = np.arange(500) * 4.0
    assert find_seabed(samples, rng.normal(size=(60, 500))) is None
    with pytest.raises(ValueError, match="traces must be"):
        find_seabed(samples, np.zeros((3, 10)))


# -- the verdict -----------------------------------------------------------------


def test_time_like_axis_with_a_known_water_depth_is_time() -> None:
    samples, traces = make_traces(4.0, 120.0)  # 120 ms = 89 m of water
    blind = check_axis(samples, traces)
    assert blind.status == "leans time"  # only the 4 ms interval points anywhere
    seabed = next(e for e in blind.evidence if e.rule == "seabed")
    assert seabed.favours is None and "can only separate" in seabed.detail
    told = check_axis(samples, traces, water_depth=89.0)
    assert told.status == "time"
    assert next(e for e in told.evidence if e.rule == "seabed").strength == "strong"


def test_depth_like_axis_with_a_known_water_depth_is_depth() -> None:
    samples, traces = make_traces(10.0, 90.0)  # 90 m of water, 10 m samples
    assert check_axis(samples, traces).status == "undetermined"  # nothing separates
    told = check_axis(samples, traces, water_depth=90.0)
    assert told.status == "depth" and told.seabed is not None


def test_a_wrong_water_depth_contradicts_both_readings() -> None:
    samples, traces = make_traces(4.0, 120.0)
    c = check_axis(samples, traces, water_depth=300.0)
    seabed = next(e for e in c.evidence if e.rule == "seabed")
    assert seabed.favours is None and "neither fits" in seabed.detail


def test_seabed_outside_the_overlap_leans_one_way() -> None:
    far, tr = make_traces(
        10.0, 600.0
    )  # 600 as depth is beyond the shelf, as time is 444 m
    assert check_axis(far, tr).status == "leans time"
    near, tr = make_traces(10.0, 70.0)  # 70 as time would be 52 m, as depth 70 m
    assert check_axis(near, tr).status == "leans depth"


def test_muted_shallow_data_are_reported_instead_of_a_seabed() -> None:
    # processed stacks often mute the shallow data; the seabed then is not in the cube.
    # Real case: the Volve full stack is blank above ~150 ms and its strongest event in
    # the top 40% is a deeper reflector, so there is no seabed to find.
    rng = np.random.default_rng(3)
    samples = np.arange(800) * 4.0
    traces = np.zeros((60, 800))
    for i in range(60):
        start = int(rng.integers(40, 60))  # data begin at 160-236 ms, varying
        body = rng.normal(0, 0.4, 800 - start)
        traces[i, start:] = body
        traces[i, 300 + int(rng.integers(-5, 5)) :: 1] *= 0.5
        traces[i] += ricker(800, 300 + int(rng.integers(-60, 60)), 2.5) * 4.0
    assert data_onset(samples, traces) == pytest.approx(200.0, abs=40.0)
    c = check_axis(samples, traces)
    assert c.seabed is None
    assert any("muted" in n for n in c.notes)
    assert "may not be in the cube" in c.report()
    # data from the first sample (no mute) give no onset note
    assert data_onset(samples, np.random.default_rng(4).normal(size=(60, 800))) is None


def test_interval_rules() -> None:
    def status(step, seabed):
        s, t = make_traces(step, seabed, n_samples=600)
        return check_axis(s, t).status

    assert status(0.01, 0.3) == "depth"  # metres in the microsecond field: strong
    assert status(4.0, 120.0) == "leans time"  # a common time interval: weak
    assert status(25.0, 200.0) == "leans depth"  # a common depth interval: weak
    assert status(10.0, 150.0) == "undetermined"  # 10 serves both, seabed fits both
    assert status(50.0, 200.0) == "undetermined"  # too coarse for either


def test_a_strong_and_a_contradicting_strong_rule_conflict() -> None:
    samples, traces = make_traces(0.01, 1.2, n_samples=600)
    # the tiny interval says depth (strong); a water depth that only time fits says time
    seabed = find_seabed(samples, traces)
    assert seabed is not None
    c = check_axis(samples, traces, water_depth=seabed.value * 0.74)
    assert c.status == "conflicting"


# -- comparing with the stated domain ---------------------------------------------


def test_against_the_stated_domain() -> None:
    samples, traces = make_traces(4.0, 120.0)
    time, depth = (
        make_domain("time", "amplitude"),
        make_domain("depth", "amplitude", vertical_unit="m"),
    )
    told = check_axis(samples, traces, water_depth=89.0)
    assert told.against(time).startswith("consistent")
    assert told.against(depth).startswith("INCONSISTENT")
    blind = check_axis(samples, traces)
    assert blind.against(time) == "consistent (weakly): the evidence leans time"
    assert blind.against(depth).startswith("leans against")
    flat = check_axis(*make_traces(10.0, 90.0))
    assert flat.status == "undetermined"
    assert flat.against(time).startswith("cannot tell")
    assert "1 ms is about 1 m" in flat.against(time)


def test_validation() -> None:
    samples, traces = make_traces(4.0, 120.0)
    with pytest.raises(ValueError, match="unknown profile"):
        check_axis(samples, traces, profile="x")
    with pytest.raises(ValueError, match="depth_unit"):
        check_axis(samples, traces, depth_unit="km")
    with pytest.raises(ValueError, match="> 0"):
        check_axis(samples, traces, water_depth=0)
    with pytest.raises(ValueError, match="at least two"):
        check_axis(np.array([0.0]), traces[:, :1])


def test_report_names_every_rule_and_says_it_is_only_a_check() -> None:
    samples, traces = make_traces(4.0, 120.0)
    text = check_axis(samples, traces, water_depth=89.0).report()
    for word in (
        "sample interval",
        "seabed",
        "base of survey",
        "Verdict: TIME",
        "plausibility",
    ):
        assert word in text
    assert isinstance(check_axis(samples, traces), DomainCheck)


# -- on a file ---------------------------------------------------------------------


def write_segy(path, samples, traces):
    spec = segyio.spec()
    spec.format = 5
    spec.samples = samples
    spec.tracecount = len(traces)
    with segyio.create(path, spec) as f:
        for i, t in enumerate(traces):
            f.header[i] = {
                segyio.TraceField.INLINE_3D: 100 + i // 10,
                segyio.TraceField.CROSSLINE_3D: 200 + i % 10,
                segyio.TraceField.CDP_X: 1000 + i,
                segyio.TraceField.CDP_Y: 5000,
                segyio.TraceField.SourceGroupScalar: 1,
                segyio.TraceField.CoordinateUnits: 1,
            }
            f.trace[i] = t.astype(np.float32)
    return path


def test_check_domain_on_a_file_reads_few_traces_and_never_sets_the_domain(
    tmp_path,
) -> None:
    samples, traces = make_traces(4.0, 120.0, n=200)
    path = write_segy(
        tmp_path / "plain.sgy", samples, traces
    )  # nothing states the domain
    with open_segy(path, cache_dir=tmp_path / "c") as src:
        assert src.domain is None
        c = src.check_domain(n_traces=50, water_depth=89.0)
        assert c.status == "time" and c.seabed is not None and c.seabed.n_traces == 50
        assert src.domain is None  # unchanged: the check never decides

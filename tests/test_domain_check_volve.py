"""The time-vs-depth check on real traces, and on a depth-stretched copy of them.

Local only. The Volve stack (Volve Terms and Conditions, based on CC BY 4.0, Equinor
and partners; credit, never sold) is NOT in the repo: the test reads about 300 traces
from the file named by the environment variable and skips itself without it. Set
``SEISMIC_LIB_VOLVE_SEGY`` to another path if yours differs.

The file ST10010ZC11_PZ_PSDM_KIRCH_FULL_T is PSDM (pre-stack DEPTH migration) output
that was converted back to TIME (the text header says "DEPTH TO TIME CONVERSION", the
name ends in "_T"). So a migration done in the depth domain can still be delivered in
time, and the word PSDM alone settles nothing about the axis. The "depth copy" below is
the same traces resampled onto a depth axis, which is what a depth-domain delivery of
them would hold.
"""

import os
import re
from pathlib import Path

import numpy as np
import pytest

from ep_seismic.volume import check_axis, make_domain, open_segy

VOLVE = Path(os.environ.get("SEISMIC_LIB_VOLVE_SEGY", ""))
pytestmark = pytest.mark.skipif(
    not os.environ.get("SEISMIC_LIB_VOLVE_SEGY") or not VOLVE.is_file(), reason="set SEISMIC_LIB_VOLVE_SEGY to the local Volve stack (not in the repo)"
)

TIME = make_domain("time", "amplitude")
DEPTH = make_domain("depth", "amplitude", vertical_unit="m")


@pytest.fixture(scope="module")
def volve(tmp_path_factory):
    with open_segy(VOLVE, cache_dir=tmp_path_factory.mktemp("c")) as src:
        pick = np.unique(np.linspace(0, src.n_traces - 1, 300).astype(int))
        yield src.samples.copy(), src.read_traces(pick), src.detect()


def to_depth(t_ms, traces, dz, v0=1700.0, k=0.5):
    """Resample time traces onto a depth axis with v(z) = v0 + k z (a typical shelf
    trend): z = (v0 / k) (exp(k t / 2) - 1), t in seconds. Linear interpolation."""
    z_of_t = (v0 / k) * (np.exp(k * (t_ms / 1000.0) / 2.0) - 1.0)
    z = np.arange(0.0, z_of_t[-1], dz)
    return z, np.array([np.interp(z, z_of_t, tr) for tr in traces]), z_of_t


def test_the_name_alone_does_not_say_depth(volve) -> None:
    _, _, detection = volve
    assert detection.vertical is None  # "PSDM" is a hint, not an answer
    assert any("PSDM" in h for h in detection.hints)


def test_the_real_time_cube_leans_time_and_does_not_claim_more(volve) -> None:
    t, traces, _ = volve
    c = check_axis(t, traces)
    assert c.status == "leans time"  # the 4 ms interval is the only thing pointing
    assert c.against(TIME).startswith("consistent (weakly)")
    assert c.against(DEPTH).startswith("leans against")
    # the shallow data are muted, so there is no seabed to use, said out loud
    assert c.seabed is None and any("blank above" in n for n in c.notes)
    # a known water depth cannot help when the seabed is not in the cube
    assert check_axis(t, traces, water_depth=80.0).status == "leans time"


def test_a_depth_copy_is_not_called_time(volve) -> None:
    t, traces, _ = volve
    # 10 m samples (a common PSDM interval): nothing separates the readings, so the
    # check must say it cannot tell and must not lean to time just because the numbers
    # are alike
    z, depth_traces, z_of_t = to_depth(t, traces, 10.0)
    c = check_axis(z, depth_traces)
    assert c.status == "undetermined"
    assert c.against(DEPTH).startswith("cannot tell")
    assert c.against(TIME).startswith("cannot tell")
    assert not any(e.favours == "time" for e in c.evidence)
    # the mute is still seen, now in metres: the data start where 204 ms maps to
    assert c.seabed is None
    (note,) = [n for n in c.notes if "blank above" in n]
    reported = float(re.search(r"blank above about ([\d.]+)", note).group(1))
    expected = float(np.interp(204.0, t, z_of_t))  # where the 204 ms onset maps to
    assert reported == pytest.approx(expected, rel=0.3)


def test_a_depth_copy_at_12_5_m_leans_depth(volve) -> None:
    t, traces, _ = volve
    z, depth_traces, _ = to_depth(t, traces, 12.5)
    c = check_axis(z, depth_traces)
    assert c.status == "leans depth"  # 12.5 is a depth interval, not a time one
    assert c.against(DEPTH).startswith("consistent (weakly)")
    assert c.against(TIME).startswith("leans against")
    # never stronger than "leans": nothing decisive is in these traces
    assert all(e.strength != "strong" for e in c.evidence)


def test_the_same_traces_are_judged_by_their_axis_not_their_look(volve) -> None:
    t, traces, _ = volve
    z, depth_traces, _ = to_depth(t, traces, 12.5)
    # identical samples, two axes: only the axis (interval) changes the verdict
    as_time = check_axis(np.arange(depth_traces.shape[1]) * 4.0, depth_traces)
    as_depth = check_axis(z, depth_traces)
    assert as_time.status == "leans time" and as_depth.status == "leans depth"

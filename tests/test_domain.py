import pytest

from ep_seismic.volume import (
    DomainUnspecified,
    SampleDomain,
    detect_domain,
    make_domain,
    open_segy,
)
from ep_seismic.volume.domain import domain_from_detection


def test_detect_from_file_name() -> None:
    d = detect_domain("D:/x/Survey_PSTM_full_stack.sgy")
    assert (d.vertical, d.quantity) == ("time", "amplitude")
    assert "file name" in d.vertical_evidence


def test_psdm_in_a_file_name_is_only_a_hint() -> None:
    # Real case: the Volve stack ST10010ZC11_PZ_PSDM_KIRCH_FULL_T is PSDM output
    # converted back to TIME ("_T", "DEPTH TO TIME CONVERSION"). The word PSDM names the
    # migration, not the axis, and used to be read as a decisive "depth".
    d = detect_domain(
        "D:/Seismic/ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.segy"
    )
    assert d.vertical is None
    assert any("PSDM" in h and "may still be in time" in h for h in d.hints)
    assert d.quantity == "amplitude"  # 'stack' still counts for the quantity


def test_folder_name_is_only_a_hint() -> None:
    # Real case: Volve folder "08.VSP_VELOCITY" holds raw-stack amplitude traces.
    d = detect_domain("D:/Well/08.VSP_VELOCITY/VSPNI_COMPUTED_1.SEGY")
    assert d.quantity is None and d.vertical is None and not d.complete
    assert any("folder name suggests quantity = velocity" in h for h in d.hints)


def test_conflicting_evidence_is_not_resolved() -> None:
    d = detect_domain("time_to_depth_converted_stack.sgy")
    assert d.vertical is None
    assert any("conflicting vertical" in n for n in d.notes)
    assert d.quantity == "amplitude"


def test_words_must_match_whole_tokens() -> None:
    assert detect_domain("overtime_depthcharge.sgy").vertical is None


def test_bare_text_header_mentions_are_only_hints() -> None:
    d = detect_domain("a.sgy", "ACQUIRED  TIME 10:30  DATE 2014-01-01")
    assert d.vertical is None and d.hints
    # Real case: a Volve VSP header labels a trace-header byte "TVDSS" while the
    # samples are time. A bare mention must never settle the domain.
    d = detect_domain("a.sgy", "MD: BYTE 41-44, TVDSS: BYTE 53-56, CDP: BYTE 5-8")
    assert d.vertical is None
    assert any("tvdss" in h for h in d.hints)


def test_keyed_text_header_lines_are_decisive() -> None:
    d = detect_domain("a.sgy", "C5 VERTICAL DOMAIN: TWT   DATA TYPE: AMPLITUDE")
    assert (d.vertical, d.quantity) == ("time", "amplitude")
    assert "text header line" in d.vertical_evidence
    d = detect_domain("a.sgy", "Z AXIS = DEPTH\nQUANTITY: VELOCITY")
    assert (d.vertical, d.quantity) == ("depth", "velocity")
    assert detect_domain("a.sgy", "VERTICAL DOMAIN: SPACE").vertical is None


def test_keyed_header_conflicting_with_file_name_is_not_resolved() -> None:
    d = detect_domain("survey_twt.sgy", "VERTICAL DOMAIN: DEPTH")
    assert d.vertical is None
    assert any("conflicting vertical" in n for n in d.notes)


def test_make_domain_units(make_segy) -> None:
    t = make_domain("time", "amplitude")
    assert (t.vertical_unit, t.quantity_unit) == ("ms", "unitless")
    d = make_domain("depth", "velocity", measurement_system=1)
    assert (d.vertical_unit, d.quantity_unit) == ("m", "m/s")
    f = make_domain("depth", "velocity", measurement_system=2)
    assert (f.vertical_unit, f.quantity_unit) == ("ft", "ft/s")
    with pytest.raises(DomainUnspecified, match="depth unit"):
        make_domain("depth", "amplitude")
    with pytest.raises(DomainUnspecified, match="velocity unit"):
        make_domain("time", "velocity")
    with pytest.raises(DomainUnspecified, match="quantity_unit"):
        make_domain("time", "density")
    ok = make_domain(
        "depth", "other", vertical_unit="m", quantity_unit="%", quantity_name="porosity"
    )
    assert "Porosity [%]" == ok.quantity_label


def test_invalid_domains_are_rejected() -> None:
    with pytest.raises(ValueError, match="vertical"):
        SampleDomain("space", "m", "amplitude", "unitless", "user")
    with pytest.raises(ValueError, match="unit"):
        SampleDomain("time", "m", "amplitude", "unitless", "user")
    with pytest.raises(ValueError, match="quantity_name"):
        SampleDomain("time", "ms", "other", "unitless", "user")
    with pytest.raises(ValueError, match="quantity"):
        SampleDomain("time", "ms", "colour", "unitless", "user")


def test_labels_and_description() -> None:
    d = make_domain(
        "depth", "velocity", measurement_system=1, reference="TVDSS", evidence="x"
    )
    assert d.vertical_label == "Depth TVDSS [m]"
    assert d.quantity_label == "Velocity [m/s]"
    assert "user: x" in d.describe()


def test_unknown_file_requires_the_user_to_say(make_segy, tmp_path) -> None:
    with open_segy(make_segy(), cache_dir=tmp_path / "c") as src:
        assert src.domain is None
        with pytest.raises(DomainUnspecified, match="set_domain"):
            src.require_domain()


def test_partial_detection_is_reported_in_the_error(make_segy, tmp_path) -> None:
    path = make_segy("survey_twt.sgy")
    with open_segy(path, cache_dir=tmp_path / "c") as src:
        with pytest.raises(DomainUnspecified, match="vertical domain = time"):
            src.require_domain()


def test_complete_detection_from_name_and_measurement_system(
    make_segy, tmp_path
) -> None:
    path = make_segy("area_depth_stack.sgy")
    with open_segy(path, cache_dir=tmp_path / "c") as src:
        d = src.require_domain()
    assert (d.vertical, d.vertical_unit, d.quantity) == ("depth", "m", "amplitude")
    assert d.origin == "detected"


def test_detection_from_text_header(make_segy, tmp_path) -> None:
    path = make_segy("a.sgy", text="VERTICAL DOMAIN: TWT  DATA TYPE: AMPLITUDE")
    with open_segy(path, cache_dir=tmp_path / "c") as src:
        d = src.require_domain()
    assert (d.vertical, d.quantity) == ("time", "amplitude")
    assert "text header" in d.evidence


def test_bare_header_mention_does_not_settle_but_is_reported(
    make_segy, tmp_path
) -> None:
    path = make_segy("a.sgy", text="TVDSS: BYTE 53-56  VELOCITY MODEL")
    with open_segy(path, cache_dir=tmp_path / "c") as src:
        assert src.domain is None
        with pytest.raises(DomainUnspecified, match="Hints only, NOT decisive"):
            src.require_domain()


def test_explicit_domain_wins_over_detection(make_segy, tmp_path) -> None:
    mine = make_domain("depth", "amplitude", vertical_unit="m")
    with open_segy(
        make_segy("x_twt_stack.sgy"), domain=mine, cache_dir=tmp_path / "c"
    ) as src:
        assert src.require_domain() is mine


def test_set_domain_is_remembered_across_sessions(make_segy, tmp_path) -> None:
    path, cache = make_segy(), tmp_path / "c"
    with open_segy(path, cache_dir=cache) as src:
        src.set_domain(make_domain("time", "amplitude", reference="MSL"))
    with open_segy(path, cache_dir=cache) as src2:
        d = src2.require_domain()
    assert d.origin == "remembered" and d.reference == "MSL"


def test_set_domain_without_remembering(make_segy, tmp_path) -> None:
    path, cache = make_segy(), tmp_path / "c"
    with open_segy(path, cache_dir=cache) as src:
        src.set_domain(make_domain("time", "amplitude"), remember=False)
        assert src.require_domain().origin == "user"
    with open_segy(path, cache_dir=cache) as src2:
        assert src2.domain is None


def test_corrupt_remembered_domain_is_ignored(make_segy, tmp_path) -> None:
    path, cache = make_segy(), tmp_path / "c"
    with open_segy(path, cache_dir=cache) as src:
        sidecar = src._domain_sidecar()
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text("{not json", encoding="utf-8")
        assert src.domain is None


def test_domain_from_detection_needs_both_parts_and_a_unit() -> None:
    assert domain_from_detection(detect_domain("a_twt.sgy")) is None
    assert (
        domain_from_detection(detect_domain("a_depth_stack.sgy"), 0) is None
    )  # no unit
    assert (
        domain_from_detection(detect_domain("a_depth_stack.sgy"), 2).vertical_unit
        == "ft"
    )

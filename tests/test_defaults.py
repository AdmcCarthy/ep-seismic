"""Every packaged default loads as a YAML mapping."""

from pathlib import Path

import yaml

DEFAULTS = list((Path(__file__).resolve().parents[1] / "src").glob("*/defaults/*.yaml"))


def test_every_default_is_a_mapping():
    for f in DEFAULTS:
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        assert isinstance(data, dict), f.name

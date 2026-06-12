"""Arg-parsing tests for the spine CLI (CLAUDE.md: test argument parsing)."""

from __future__ import annotations

from spine.cli import _hp_from_extras, _parse_known


def test_hp_from_extras_key_value_and_flags() -> None:
    hp = _hp_from_extras(["--feat", "graph", "--depth", "deep", "--fast"])
    assert hp == {"feat": "graph", "depth": "deep", "fast": True}


def test_hp_from_extras_dashes_become_underscores() -> None:
    assert _hp_from_extras(["--learning-rate", "0.01"]) == {"learning_rate": "0.01"}


def test_parse_known_splits_spine_flags_from_head_hp() -> None:
    args, extras = _parse_known(
        ["--domain", "molecular", "--max-steps", "5", "--feat", "ecfp"]
    )
    assert args.domain == "molecular"
    assert args.max_steps == 5
    assert extras == ["--feat", "ecfp"]

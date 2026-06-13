"""Shape tests for the weather head's synthetic data (CLAUDE.md: test shapes)."""

from __future__ import annotations

from heads.weather import data as wx


def test_make_fields_shapes_and_determinism() -> None:
    a = wx.make_fields(n=8, size=32, seed=1)
    b = wx.make_fields(n=8, size=32, seed=1)
    assert a.x.shape == a.y.shape == (8, 1, 32, 32)
    assert a.x.dtype.is_floating_point
    # deterministic for a fixed seed
    assert (a.x == b.x).all() and (a.y == b.y).all()


def test_make_fields_resolution_axis() -> None:
    f = wx.make_fields(n=4, size=16, seed=2)
    assert f.x.shape == (4, 1, 16, 16)


def test_input_is_degraded_truth() -> None:
    # x should differ from y (blur + noise applied), not be a copy
    f = wx.make_fields(n=4, size=32, seed=3)
    assert not (f.x == f.y).all()

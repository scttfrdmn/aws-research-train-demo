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


def test_target_is_pde_evolved_input() -> None:
    # y is x evolved one PDE step (diffusion + advective shift) — differs from x,
    # and diffusion reduces variance (energy decays), so y is not just x rolled.
    f = wx.make_fields(n=8, size=32, seed=3)
    assert not (f.x == f.y).all()
    assert f.y.var().item() <= f.x.var().item() + 1e-6

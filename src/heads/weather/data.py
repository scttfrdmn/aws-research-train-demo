"""Synthetic spatial-field data for the weather head (SPEC §3, issue #9).

A **PDE solution-operator** task, generated in code (no ERA5 download,
deterministic, seeded): given an initial field ``x``, predict the field ``y``
after one step of a known PDE — here the 2-D diffusion (heat) equation, plus a
fixed advective shift. Learning the map ``x → y`` is *operator learning* — the
problem neural operators (FNO) exist to solve in 2026 — not image restoration.

The initial field is a band-limited Gaussian random field (coherent spatial
structure an FNO/U-Net can represent); the target applies the analytic diffusion
operator in the spectral domain. The model learns to emulate the solver, and the
truth|pred|error delta is naked-eye obvious after a short train.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Fields:
    """A field-pair dataset: initial fields x → evolved fields y, shape (N,1,H,W)."""

    x: torch.Tensor
    y: torch.Tensor


def _low_pass_field(n: int, size: int, k0: float, gen: torch.Generator) -> torch.Tensor:
    """N band-limited Gaussian random fields via a radial spectral low-pass.

    White noise → rfft2 → multiply by 1/(1+(k/k0)^2) → irfft2 → unit variance.
    Produces coherent blobs (not noise) — a smooth initial condition for the PDE.
    """
    noise = torch.randn(n, 1, size, size, generator=gen)
    spec = torch.fft.rfft2(noise)
    ky = torch.fft.fftfreq(size).view(size, 1) * size
    kx = torch.fft.rfftfreq(size).view(1, size // 2 + 1) * size
    kr = torch.sqrt(ky**2 + kx**2)
    envelope = 1.0 / (1.0 + (kr / k0) ** 2)
    field: torch.Tensor = torch.fft.irfft2(spec * envelope, s=(size, size))
    field = field / field.flatten(1).std(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
    return field


def _diffuse(u: torch.Tensor, size: int, nu: float, dt: float, shift: int) -> torch.Tensor:
    """One step of the 2-D diffusion (heat) equation + a fixed advective roll.

    The analytic solution operator in Fourier space: û(t+dt) = û(t)·exp(-ν|k|²·dt).
    This is a real, known PDE operator — what the model learns to emulate.
    """
    uf = torch.fft.rfft2(u)
    ky = (torch.fft.fftfreq(size).view(size, 1) * 2 * torch.pi * size) ** 2
    kx = (torch.fft.rfftfreq(size).view(1, size // 2 + 1) * 2 * torch.pi * size) ** 2
    decay = torch.exp(-nu * (ky + kx) * dt)
    evolved: torch.Tensor = torch.fft.irfft2(uf * decay, s=(size, size))
    # a fixed advective shift, so the operator has a transport component too
    return torch.roll(evolved, shifts=shift, dims=-1)


def make_fields(
    n: int = 256,
    size: int = 32,
    k0: float = 4.0,
    nu: float = 1e-4,
    dt: float = 1.0,
    shift: int = 2,
    seed: int = 0,
) -> Fields:
    """Generate the PDE-operator dataset: y = diffuse(x) for initial fields x."""
    gen = torch.Generator().manual_seed(seed)
    x = _low_pass_field(n, size, k0, gen)
    y = _diffuse(x, size, nu, dt, shift)
    return Fields(x=x, y=y)

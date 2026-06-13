"""Synthetic spatial-field data for the weather head (SPEC §3, issue #9).

A deblur task on band-limited Gaussian random fields, generated in code (no ERA5
download, deterministic, seeded). The truth is a smooth field with coherent
spatial structure an FNO/U-Net can represent; the input is a fixed known
degradation (blur + noise). The model learns to restore high-frequency
structure — the FNO/U-Net wheelhouse — and the truth|pred|error delta is
naked-eye obvious after a short train (verify-first #9).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Fields:
    """A field-pair dataset: degraded inputs x → smooth truths y, shape (N,1,H,W)."""

    x: torch.Tensor
    y: torch.Tensor


def _low_pass_field(n: int, size: int, k0: float, gen: torch.Generator) -> torch.Tensor:
    """N band-limited Gaussian random fields via a radial spectral low-pass.

    White noise → rfft2 → multiply by 1/(1+(k/k0)^2) → irfft2 → unit variance.
    Produces coherent blobs (not noise), which is what makes the task learnable.
    """
    noise = torch.randn(n, 1, size, size, generator=gen)
    spec = torch.fft.rfft2(noise)
    # radial wavenumber grid matching the rfft2 output (size x size//2+1)
    ky = torch.fft.fftfreq(size).view(size, 1) * size
    kx = torch.fft.rfftfreq(size).view(1, size // 2 + 1) * size
    kr = torch.sqrt(ky**2 + kx**2)
    envelope = 1.0 / (1.0 + (kr / k0) ** 2)
    field: torch.Tensor = torch.fft.irfft2(spec * envelope, s=(size, size))
    # normalize each sample to ~unit variance
    field = field / field.flatten(1).std(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
    return field


def _blur(y: torch.Tensor, size: int) -> torch.Tensor:
    """A fixed, known degradation: downscale-by-2 then bilinear upscale (blur)."""
    lo = torch.nn.functional.avg_pool2d(y, kernel_size=2)
    return torch.nn.functional.interpolate(
        lo, size=(size, size), mode="bilinear", align_corners=False
    )


def make_fields(
    n: int = 256, size: int = 32, k0: float = 4.0, noise: float = 0.1, seed: int = 0
) -> Fields:
    """Generate the deblur dataset: x = blur(y) + noise, target y."""
    gen = torch.Generator().manual_seed(seed)
    y = _low_pass_field(n, size, k0, gen)
    x = _blur(y, size) + noise * torch.randn(n, 1, size, size, generator=gen)
    return Fields(x=x, y=y)

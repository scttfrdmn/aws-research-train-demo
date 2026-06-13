"""Synthetic DNA → regulatory-signal data for the genomics head (SPEC §3, #10).

Phantom-science: DNA + labels are generated in code (no real genomic data, no
licensing friction). Positives carry a fixed TF-binding motif implanted at a
random position; negatives carry a composition-matched scrambled motif. A small
1D CNN must learn a motif-matched filter to separate them — so a trained model's
input-gradient saliency sharply peaks at the implanted motif while an untrained
one is flat (the naked-eye delta, verify-first #10).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

BASES = "ACGT"
MOTIF = "GATAAG"  # a GATA-box-like consensus; the signal positives carry


@dataclass(frozen=True)
class Seqs:
    """One-hot DNA `x` (N,4,L) and binary labels `y` (N,)."""

    x: torch.Tensor
    y: torch.Tensor
    motif: str
    length: int


def _one_hot(seqs: np.ndarray) -> torch.Tensor:
    """(N, L) int codes 0..3 → (N, 4, L) float one-hot."""
    n, length = seqs.shape
    oh = np.zeros((n, 4, length), dtype=np.float32)
    rows = np.arange(n)[:, None]
    cols = np.arange(length)[None, :]
    oh[rows, seqs, cols] = 1.0
    return torch.from_numpy(oh)


def _codes(s: str) -> np.ndarray:
    return np.array([BASES.index(c) for c in s], dtype=np.int64)


def make_seqs(
    n: int = 4000, length: int = 200, motif: str = MOTIF, seed: int = 0
) -> Seqs:
    """Balanced positives/negatives; positives get `motif` at a random position,
    negatives get a scrambled (composition-matched) motif so chance matches
    don't leak the label.
    """
    rng = np.random.default_rng(seed)
    seqs = rng.integers(0, 4, size=(n, length), dtype=np.int64)
    labels = np.zeros(n, dtype=np.float32)
    m = _codes(motif)
    scrambled = m.copy()
    rng.shuffle(scrambled)
    mlen = len(m)
    for i in range(n):
        pos = int(rng.integers(0, length - mlen + 1))
        if i % 2 == 0:  # positive
            seqs[i, pos : pos + mlen] = m
            labels[i] = 1.0
        else:  # negative — equal-composition scrambled motif
            seqs[i, pos : pos + mlen] = scrambled
    return Seqs(
        x=_one_hot(seqs), y=torch.from_numpy(labels), motif=motif, length=length
    )


def encode(seq: str) -> torch.Tensor:
    """One SMILES-like DNA string → (1, 4, L) one-hot, for the viewer/predict."""
    codes = _codes(seq.upper())[None, :]
    return _one_hot(codes)

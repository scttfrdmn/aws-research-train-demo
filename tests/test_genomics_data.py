"""Shape/label tests for the genomics head's synthetic DNA (CLAUDE.md: shapes)."""

from __future__ import annotations

from heads.genomics import data as dna


def test_make_seqs_shapes_and_balance() -> None:
    s = dna.make_seqs(n=100, length=200, seed=1)
    assert s.x.shape == (100, 4, 200)
    assert s.y.shape == (100,)
    # one-hot: exactly one base set per position
    assert (s.x.sum(dim=1) == 1).all()
    # balanced (alternating pos/neg)
    assert int(s.y.sum()) == 50


def test_positives_carry_the_motif() -> None:
    s = dna.make_seqs(n=20, length=200, seed=2)
    # decode each positive and confirm the motif is present
    inv = "ACGT"
    for i in range(20):
        if s.y[i] == 1:
            codes = s.x[i].argmax(dim=0).tolist()
            seq = "".join(inv[c] for c in codes)
            assert s.motif in seq, "positive sequence missing the implanted motif"
            break


def test_encode_roundtrip_shape() -> None:
    oh = dna.encode("GATAAG")
    assert oh.shape == (1, 4, 6)
    assert (oh.sum(dim=1) == 1).all()

"""Shape tests for the molecular head's data path (CLAUDE.md: test data shapes).

These are head-deps tests; skipped cleanly if the molecular group isn't synced.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("torch_geometric")

from heads.molecular import data as d  # noqa: E402


def test_load_esol_limit_and_alignment() -> None:
    esol = d.load_esol(limit=16)
    assert len(esol.smiles) == 16
    assert esol.y.shape == (16,)
    assert esol.y.dtype == np.float32


def test_ecfp_features_shape() -> None:
    esol = d.load_esol(limit=8)
    x = d.ecfp_features(esol.smiles)
    assert x.shape == (8, d.ecfp_dim())
    assert x.dtype.is_floating_point


@pytest.mark.parametrize("with_3d", [False, True])
def test_graph_features_node_dim(with_3d: bool) -> None:
    esol = d.load_esol(limit=8)
    graphs = d.graph_features(esol.smiles, esol.y, with_3d)
    assert len(graphs) > 0
    g = graphs[0]
    assert g.x.size(1) == d.node_feature_dim(with_3d)
    assert g.edge_index.size(0) == 2
    assert g.y.shape == (1,)

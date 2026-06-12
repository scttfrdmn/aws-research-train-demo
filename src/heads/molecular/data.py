"""ESOL data + featurization for the molecular head (SPEC §3, issue #2).

Everything is built offline from the vendored ``delaney-processed.csv`` + RDKit
— no network, no DeepChem, no OGB feature dims hardcoded (verify-first #2/#6).

Two featurization families back the ``feat`` sweep axis:

- ``ecfp``        — Morgan fingerprint (generator API) → a dense vector → MLP.
- ``graph``       — per-atom features + bond edges → a small GNN (PyG core).
- ``graph+3d``    — ``graph`` plus 3D conformer coordinates as extra node feats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator

# The vendored ESOL benchmark (1128 molecules). Columns of interest:
# ``smiles`` and ``measured log solubility in mols per litre``.
_CSV = Path(__file__).parent / "data" / "delaney-processed.csv"
_SMILES_COL = "smiles"
_TARGET_COL = "measured log solubility in mols per litre"

# A small, fixed set of elements covers ESOL; anything else falls in an "other"
# bucket. Kept explicit so node-feature dims are stable and never hardcoded
# blindly (risk #6 — dims are derived from these constants, asserted at runtime).
_ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]


@dataclass(frozen=True)
class Esol:
    """A loaded ESOL split: parallel SMILES + target arrays."""

    smiles: list[str]
    y: np.ndarray  # measured log-solubility, shape (N,)


def load_esol(csv_path: str | Path = _CSV, limit: int | None = None) -> Esol:
    """Load ESOL from the vendored CSV. ``limit`` takes the first N rows (smoke).

    Uses pandas if available, else a stdlib csv fallback, so the smoke path
    doesn't hard-depend on the import ordering of the head group.
    """
    import csv

    smiles: list[str] = []
    targets: list[float] = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = row[_SMILES_COL].strip()
            if not smi:
                continue
            smiles.append(smi)
            targets.append(float(row[_TARGET_COL]))
            if limit is not None and len(smiles) >= limit:
                break
    return Esol(smiles=smiles, y=np.asarray(targets, dtype=np.float32))


# --- ECFP -----------------------------------------------------------------

_FP_SIZE = 2048
_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=_FP_SIZE)


def ecfp_features(smiles: list[str]) -> torch.Tensor:
    """Morgan fingerprints via the current generator API → (N, 2048) float."""
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(_FP_SIZE, dtype=np.uint8))
            continue
        rows.append(_morgan.GetFingerprintAsNumPy(mol))
    return torch.from_numpy(np.stack(rows)).float()


def ecfp_dim() -> int:
    return _FP_SIZE


# --- Graph ----------------------------------------------------------------


def _atom_features(atom: Chem.Atom) -> list[float]:
    """A small, stable per-atom feature vector."""
    sym = atom.GetSymbol()
    onehot = [1.0 if sym == e else 0.0 for e in _ELEMENTS]
    onehot.append(0.0 if sym in _ELEMENTS else 1.0)  # "other" bucket
    return onehot + [
        float(atom.GetDegree()),
        float(atom.GetTotalNumHs()),
        float(atom.GetFormalCharge()),
        1.0 if atom.GetIsAromatic() else 0.0,
    ]


def node_feature_dim(with_3d: bool) -> int:
    base = len(_ELEMENTS) + 1 + 4  # elements + other + 4 scalar feats
    return base + (3 if with_3d else 0)


def graph_features(smiles: list[str], y: np.ndarray, with_3d: bool) -> list[Any]:
    """Build PyG ``Data`` objects (atoms→nodes, bonds→undirected edges).

    ``with_3d`` appends embedded conformer xyz coordinates to each node.
    Returns a list of ``torch_geometric.data.Data``.
    """
    from torch_geometric.data import Data

    graphs: list[Any] = []
    for smi, target in zip(smiles, y, strict=True):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        coords = None
        if with_3d:
            mol3d = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol3d, randomSeed=0xF00D) == 0:
                AllChem.MMFFOptimizeMolecule(mol3d, maxIters=200)
                conf = mol3d.GetConformer()
                # map back to heavy-atom order (mol, not mol3d-with-Hs)
                coords = np.array(
                    [list(conf.GetAtomPosition(a.GetIdx())) for a in mol.GetAtoms()],
                    dtype=np.float32,
                )

        feats = [_atom_features(a) for a in mol.GetAtoms()]
        x = torch.tensor(feats, dtype=torch.float)
        if with_3d:
            xyz = (
                torch.from_numpy(coords)
                if coords is not None and len(coords) == x.size(0)
                else torch.zeros((x.size(0), 3))
            )
            x = torch.cat([x, xyz], dim=1)

        edges = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges += [(i, j), (j, i)]
        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        graphs.append(
            Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor([float(target)], dtype=torch.float),
            )
        )
    return graphs

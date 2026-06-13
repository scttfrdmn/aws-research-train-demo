"""Molecular head — ESOL log-solubility regression (SPEC §3, issue #7).

The reference head: tiny models, minutes to train, license-clean, legible
output. Implements the ``Head`` contract; the spine never special-cases it.

Sweep axis (the hypothesis): featurization ``{ecfp, graph, graph+3d}`` × depth
``{shallow, deep}`` — scientific choices, not CS knobs. Metric: RMSE.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from heads.base import SweepAxis
from heads.molecular import data as mol_data

if TYPE_CHECKING:
    from heads.base import Viewer
    from spine.run import Run

# log-solubility is unitless small floats; a fixed seed keeps smoke deterministic.
_SEED = 0xF00D


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _Axis:
    """Concrete SweepAxis (the contract declares it structurally)."""

    def __init__(self, name: str, values: list[str]) -> None:
        self.name = name
        self.values = values


class _MLP(nn.Module):
    """ECFP → scalar. ``deep`` stacks more hidden layers than ``shallow``."""

    def __init__(self, in_dim: int, depth: str) -> None:
        super().__init__()
        widths = [512, 256, 128] if depth == "deep" else [256]
        layers: list[nn.Module] = []
        d = in_dim
        for w in widths:
            layers += [nn.Linear(d, w), nn.ReLU()]
            d = w
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x).squeeze(-1)
        return out


class _GNN(nn.Module):
    """Graph → scalar via mean-pooled GCN layers. ``deep`` adds more layers."""

    def __init__(self, in_dim: int, depth: str) -> None:
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool

        hidden = 128
        n_layers = 4 if depth == "deep" else 2
        self.convs = nn.ModuleList()
        d = in_dim
        for _ in range(n_layers):
            self.convs.append(GCNConv(d, hidden))
            d = hidden
        self.head = nn.Linear(hidden, 1)
        self._pool = global_mean_pool

    def forward(self, batch: Any) -> torch.Tensor:
        x, edge_index = batch.x, batch.edge_index
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        out: torch.Tensor = self.head(self._pool(x, batch.batch)).squeeze(-1)
        return out


class MolecularHead:
    """The molecular domain head (see module docstring)."""

    name = "molecular"
    dependency_group = "molecular"
    _ckpt_every = 25  # epochs between checkpoints (spot-resume granularity)
    _epoch_delay = 0.0  # seconds slept between epochs (--epoch-delay pacing knob)

    # --- contract ---------------------------------------------------------

    def prepare_data(self, data_dir: str, split: str) -> str:
        """Return the dataset path. The vendored ESOL CSV is the local sample;
        cloud runs pass an S3-backed ``data_dir`` (resolved by the submitter).
        """
        candidate = Path(data_dir) / "delaney-processed.csv"
        if candidate.exists():
            return str(candidate)
        return str(mol_data._CSV)  # vendored fallback for smoke

    def sweep_axes(self) -> list[SweepAxis]:
        return [
            _Axis("feat", ["ecfp", "graph", "graph+3d"]),
            _Axis("depth", ["shallow", "deep"]),
        ]

    def tile_label(self, hp: dict[str, Any]) -> str:
        # " / " separator (not "·"): the label becomes a SageMaker tag value,
        # whose charset is [\p{L}\p{Z}\p{N}_.:/=+\-@] — middle-dot is rejected.
        return f"feat={hp.get('feat', 'ecfp')} / depth={hp.get('depth', 'shallow')}"

    def metric_name(self) -> str:
        return "rmse"

    def fit(self, run: Run, hp: dict[str, Any]) -> None:
        torch.manual_seed(_SEED)
        feat = str(hp.get("feat", "ecfp"))
        depth = str(hp.get("depth", "shallow"))
        epochs = int(hp.get("epochs", 50))
        lr = float(hp.get("lr", 1e-3))
        # Pacing knob: seconds to sleep between epochs. ESOL is so small that a
        # run finishes in seconds — useless for watching a sweep or interrupting
        # a spot job mid-stream. --epoch-delay stretches wall-clock without
        # changing the science (it's a CLI arg, not a constant).
        self._epoch_delay = float(hp.get("epoch_delay", 0.0))

        # Smoke mode (--max-steps) trains a small subset so even graph+3d's
        # conformer embedding stays in seconds; full runs load all of ESOL.
        limit = 128 if run.max_steps is not None else None
        esol = mol_data.load_esol(self.prepare_data(run.data_dir, "train"), limit=limit)
        device = _device()

        if feat == "ecfp":
            self._fit_ecfp(run, esol, depth, epochs, lr, device)
        else:
            self._fit_graph(run, esol, feat, depth, epochs, lr, device)

    def predict(self, checkpoint: str, x: Any) -> float:
        """Predict log-solubility for one SMILES string ``x``."""
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        feat, depth = ckpt["feat"], ckpt["depth"]
        model: nn.Module
        if feat == "ecfp":
            model = _MLP(mol_data.ecfp_dim(), depth)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            with torch.no_grad():
                return float(model(mol_data.ecfp_features([x]))[0])
        from torch_geometric.loader import DataLoader

        with_3d = feat == "graph+3d"
        model = _GNN(mol_data.node_feature_dim(with_3d), depth)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        graphs = mol_data.graph_features([x], np.zeros(1, dtype=np.float32), with_3d)
        with torch.no_grad():
            batch = next(iter(DataLoader(graphs, batch_size=1)))
            return float(model(batch)[0])

    def viewer(self) -> Viewer:
        from heads.molecular.viewer import MolecularViewer

        return MolecularViewer()

    # --- training internals ----------------------------------------------

    def _max_epochs(self, run: Run, epochs: int) -> int:
        """Smoke (`--max-steps`) caps epochs so a real fwd/bwd runs in seconds."""
        if run.max_steps is not None:
            return max(1, min(epochs, run.max_steps))
        return epochs

    def _checkpoint(
        self,
        run: Run,
        model: nn.Module,
        feat: str,
        depth: str,
        epoch: int = 0,
        opt: torch.optim.Optimizer | None = None,
    ) -> None:
        """Persist model + optimizer + epoch to the (S3-synced) checkpoint dir.

        Including the optimizer state and last epoch is what lets a spot-reclaimed
        job *resume* mid-training rather than restart (stage 5, SPEC §4).
        """
        Path(run.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        path = Path(run.checkpoint_dir) / "model.pt"
        payload = {
            "state_dict": model.state_dict(),
            "feat": feat,
            "depth": depth,
            "epoch": epoch,
        }
        if opt is not None:
            payload["optimizer"] = opt.state_dict()
        torch.save(payload, path)
        (Path(run.checkpoint_dir) / "meta.json").write_text(
            json.dumps(
                {
                    "feat": feat,
                    "depth": depth,
                    "metric": self.metric_name(),
                    "epoch": epoch,
                }
            )
        )

    def _resume(
        self,
        run: Run,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        feat: str,
    ) -> int:
        """Resume from an existing checkpoint if one matches; return start epoch.

        On a spot restart SageMaker re-syncs the checkpoint S3 URI into the
        container, so an interrupted job sees its last `model.pt` here and picks
        up where it left off. Returns 0 for a fresh run.
        """
        path = Path(run.checkpoint_dir) / "model.pt"
        if not path.exists():
            return 0
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("feat") != feat:  # different sweep point — ignore
            return 0
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start = int(ckpt.get("epoch", 0)) + 1
        run.metric_sink.log(start, {"resumed_from_epoch": float(ckpt.get("epoch", 0))})
        return start

    def _fit_ecfp(
        self,
        run: Run,
        esol: mol_data.Esol,
        depth: str,
        epochs: int,
        lr: float,
        dev: torch.device,
    ) -> None:
        x = mol_data.ecfp_features(esol.smiles).to(dev)
        y = torch.from_numpy(esol.y).to(dev)
        model = _MLP(mol_data.ecfp_dim(), depth).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        total = self._max_epochs(run, epochs)
        start = self._resume(run, model, opt, "ecfp")
        for epoch in range(start, total):
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            rmse = float(torch.sqrt(loss.detach()))
            run.metric_sink.log(epoch, {self.metric_name(): rmse})
            if epoch % self._ckpt_every == 0:
                self._checkpoint(run, model, "ecfp", depth, epoch, opt)
            if self._epoch_delay:
                time.sleep(self._epoch_delay)
        self._checkpoint(run, model, "ecfp", depth, total - 1, opt)

    def _fit_graph(
        self,
        run: Run,
        esol: mol_data.Esol,
        feat: str,
        depth: str,
        epochs: int,
        lr: float,
        dev: torch.device,
    ) -> None:
        from torch_geometric.loader import DataLoader

        with_3d = feat == "graph+3d"
        graphs = mol_data.graph_features(esol.smiles, esol.y, with_3d)
        loader = DataLoader(graphs, batch_size=64, shuffle=True)
        model = _GNN(mol_data.node_feature_dim(with_3d), depth).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        total = self._max_epochs(run, epochs)
        start = self._resume(run, model, opt, feat)
        for epoch in range(start, total):
            model.train()
            sq_err, n = 0.0, 0
            for batch in loader:
                batch = batch.to(dev)
                opt.zero_grad()
                pred = model(batch)
                loss = loss_fn(pred, batch.y)
                loss.backward()
                opt.step()
                sq_err += float(loss.detach()) * batch.num_graphs
                n += batch.num_graphs
            rmse = float(np.sqrt(sq_err / max(n, 1)))
            run.metric_sink.log(epoch, {self.metric_name(): rmse})
            if epoch % self._ckpt_every == 0:
                self._checkpoint(run, model, feat, depth, epoch, opt)
            if self._epoch_delay:
                time.sleep(self._epoch_delay)
        self._checkpoint(run, model, feat, depth, total - 1, opt)


# The registry loads this attribute (see spine/registry.py).
HEAD = MolecularHead()

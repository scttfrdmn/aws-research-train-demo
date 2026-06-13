"""Genomics head — DNA → regulatory signal (SPEC §3, issue #10).

Detect an implanted motif. Sweep axis receptive-field {small, large} × arch
{cnn, dilated}. Metric auROC. Implements the contract; the spine never
special-cases it. Plain 1D CNN in torch+numpy — no genomics library (#10).
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
from heads.genomics import data as dna

if TYPE_CHECKING:
    from heads.base import Viewer
    from spine.run import Run

_SEED = 0xF00D


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _Axis:
    def __init__(self, name: str, values: list[str]) -> None:
        self.name = name
        self.values = values


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Hand-rolled auROC (rank statistic) — keeps the metric torch+numpy-pure."""
    s = scores.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


class _DNACNN(nn.Module):
    """1D CNN over one-hot DNA. The sweep axes set kernel/depth (receptive field)
    and dilation (cnn vs dilated). Global max-pool → translation invariance.
    """

    def __init__(self, receptive: str, arch: str, channels: int = 32) -> None:
        super().__init__()
        n_layers = 3 if receptive == "large" else 1
        kernel = 7
        dilations = (
            [2**i for i in range(n_layers)] if arch == "dilated" else [1] * n_layers
        )
        layers: list[nn.Module] = []
        in_ch = 4
        for d in dilations:
            pad = (kernel - 1) * d // 2
            layers += [
                nn.Conv1d(in_ch, channels, kernel, dilation=d, padding=pad),
                nn.ReLU(),
            ]
            in_ch = channels
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        pooled = h.max(dim=-1).values  # global max-pool over length
        out: torch.Tensor = self.head(pooled).squeeze(-1)
        return out


class GenomicsHead:
    """The genomics domain head (see module docstring)."""

    name = "genomics"
    dependency_group = "genomics"
    _ckpt_every = 25
    _epoch_delay = 0.0

    def prepare_data(self, data_dir: str, split: str) -> str:
        return data_dir  # generated in code

    def sweep_axes(self) -> list[SweepAxis]:
        return [
            _Axis("receptive", ["small", "large"]),
            _Axis("arch", ["cnn", "dilated"]),
        ]

    def tile_label(self, hp: dict[str, Any]) -> str:
        return f"rf={hp.get('receptive', 'small')} / arch={hp.get('arch', 'cnn')}"

    def metric_name(self) -> str:
        return "auroc"

    def fit(self, run: Run, hp: dict[str, Any]) -> None:
        torch.manual_seed(_SEED)
        receptive = str(hp.get("receptive", "small"))
        arch = str(hp.get("arch", "cnn"))
        epochs = int(hp.get("epochs", 150))
        lr = float(hp.get("lr", 1e-3))
        self._epoch_delay = float(hp.get("epoch_delay", 0.0))

        n = 256 if run.max_steps is not None else 4000
        length = 200
        seqs = dna.make_seqs(n=n, length=length)
        dev = _device()
        x, y = seqs.x.to(dev), seqs.y.to(dev)
        model = _DNACNN(receptive, arch).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        total = max(1, min(epochs, run.max_steps)) if run.max_steps else epochs
        start = self._resume(run, model, opt, receptive, arch)
        for epoch in range(start, total):
            model.train()
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            metric = auroc(torch.sigmoid(logits), y)
            run.metric_sink.log(epoch, {self.metric_name(): metric})
            if epoch % self._ckpt_every == 0:
                self._checkpoint(run, model, receptive, arch, epoch, opt, total, metric)
            if self._epoch_delay:
                time.sleep(self._epoch_delay)
        self._checkpoint(run, model, receptive, arch, total - 1, opt, total, metric)

    def predict(self, checkpoint: str, x: Any) -> Any:
        """Per-position saliency (input-gradient) for one DNA string `x`.

        Returns a length-L numpy track that peaks at the implanted motif once
        trained, flat when untrained — the viewer's saliency signal.
        """
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = _DNACNN(ckpt["receptive"], ckpt["arch"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        oh = (
            dna.encode(x)
            if isinstance(x, str)
            else torch.as_tensor(x, dtype=torch.float)
        )
        while oh.dim() < 3:
            oh = oh.unsqueeze(0)
        oh.requires_grad_(True)
        logit = model(oh)[0]
        logit.backward()
        grad = oh.grad
        assert grad is not None
        # per-position importance = grad×input on the present base (sum channels)
        sal = (grad * oh.detach()).sum(dim=1)[0]
        return sal.detach().cpu().numpy()

    def viewer(self) -> Viewer:
        from heads.genomics.viewer import GenomicsViewer

        return GenomicsViewer()

    # --- internals -------------------------------------------------------

    def _checkpoint(
        self,
        run: Run,
        model: nn.Module,
        receptive: str,
        arch: str,
        epoch: int,
        opt: torch.optim.Optimizer,
        total: int,
        metric: float,
    ) -> None:
        Path(run.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "receptive": receptive,
                "arch": arch,
                "epoch": epoch,
                "optimizer": opt.state_dict(),
            },
            Path(run.checkpoint_dir) / "model.pt",
        )
        (Path(run.checkpoint_dir) / "meta.json").write_text(
            json.dumps(
                {
                    "receptive": receptive,
                    "arch": arch,
                    "metric": self.metric_name(),
                    "epoch": epoch,
                    "total": total,
                    "auroc": metric,
                }
            )
        )

    def _resume(
        self,
        run: Run,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        receptive: str,
        arch: str,
    ) -> int:
        path = Path(run.checkpoint_dir) / "model.pt"
        if not path.exists():
            return 0
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("receptive") != receptive or ckpt.get("arch") != arch:
            return 0
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        return int(ckpt.get("epoch", 0)) + 1


# The registry loads this attribute (see spine/registry.py).
HEAD = GenomicsHead()

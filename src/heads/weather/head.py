"""Weather head — spatial field prediction (SPEC §3, issue #9).

**Operator learning:** given an initial field, predict the field after one step
of a known PDE (2-D diffusion + advection) — the map a neural operator is built
to emulate in 2026, not image restoration. Sweep axis operator {fno, unet} ×
resolution. Metric: field RMSE. Implements the contract; the spine never
special-cases it.

Both operators are hand-rolled in plain torch (verify-first #9): a tiny FNO
(SpectralConv2d via torch.fft, differentiable on CPU) and a tiny U-Net. No
neuraloperator dependency — zero extra core deps.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from heads.base import SweepAxis
from heads.weather import data as wx_data

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


class _SpectralConv2d(nn.Module):
    """The FNO core: rfft2 → keep low modes → complex weight → irfft2.

    A complex weight per retained (ky, kx) mode pair; Adam trains it unmodified
    (complex autograd, CPU-fine on torch 2.12 — verify-first #9).
    """

    def __init__(self, in_ch: int, out_ch: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * out_ch)
        # two blocks of modes (low + wrapped-high in ky), as in the canonical FNO
        self.w1 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes, modes, dtype=torch.cfloat)
        )
        self.w2 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        m = min(self.modes, h, w // 2 + 1)
        xf = torch.fft.rfft2(x)
        out = torch.zeros(
            b, self.w1.shape[1], h, w // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        out[:, :, :m, :m] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, :m, :m], self.w1[:, :, :m, :m]
        )
        out[:, :, -m:, :m] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, -m:, :m], self.w2[:, :, :m, :m]
        )
        inv: torch.Tensor = torch.fft.irfft2(out, s=(h, w))
        return inv


class _FNO(nn.Module):
    """Lift → stacked spectral+pointwise blocks → project. Resolution-agnostic."""

    def __init__(self, width: int = 24, modes: int = 10, depth: int = 3) -> None:
        super().__init__()
        self.lift = nn.Conv2d(1, width, 1)
        self.spectral = nn.ModuleList(
            [_SpectralConv2d(width, width, modes) for _ in range(depth)]
        )
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.project = nn.Sequential(
            nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for sp, lc in zip(self.spectral, self.local, strict=True):
            x = x + torch.relu(sp(x) + lc(x))
        out: torch.Tensor = self.project(x)
        return out


class _UNet(nn.Module):
    """Tiny 2-down / 2-up U-Net — the local-conv contrast to the FNO."""

    def __init__(self, base: int = 16) -> None:
        super().__init__()

        def block(i: int, o: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(o, o, 3, padding=1),
                nn.GELU(),
            )

        self.d1 = block(1, base)
        self.d2 = block(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.mid = block(base * 2, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.u2 = block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = block(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        m = self.mid(self.pool(c2))
        u2 = self.u2(torch.cat([self.up2(m), c2], dim=1))
        u1 = self.u1(torch.cat([self.up1(u2), c1], dim=1))
        out: torch.Tensor = self.out(u1)
        return out


def _build(operator: str) -> nn.Module:
    return _UNet() if operator == "unet" else _FNO()


class WeatherHead:
    """The weather domain head (see module docstring)."""

    name = "weather"
    dependency_group = "weather"
    _ckpt_every = 25
    _epoch_delay = 0.0

    def prepare_data(self, data_dir: str, split: str) -> str:
        # Data is generated in code; data_dir is unused for the local sample.
        return data_dir

    def sweep_axes(self) -> list[SweepAxis]:
        return [
            _Axis("operator", ["fno", "unet"]),
            _Axis("resolution", ["32", "16"]),
        ]

    def tile_label(self, hp: dict[str, Any]) -> str:
        return f"op={hp.get('operator', 'fno')} / res={hp.get('resolution', '32')}"

    def metric_name(self) -> str:
        return "rmse"

    def fit(self, run: Run, hp: dict[str, Any]) -> None:
        torch.manual_seed(_SEED)
        operator = str(hp.get("operator", "fno"))
        size = int(hp.get("resolution", 32))
        epochs = int(hp.get("epochs", 200))
        lr = float(hp.get("lr", 1e-3))
        self._epoch_delay = float(hp.get("epoch_delay", 0.0))

        n = 64 if run.max_steps is not None else 256
        fields = wx_data.make_fields(n=n, size=size)
        dev = _device()
        x, y = fields.x.to(dev), fields.y.to(dev)
        model = _build(operator).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        total = max(1, min(epochs, run.max_steps)) if run.max_steps else epochs
        start = self._resume(run, model, opt, operator, size)
        for epoch in range(start, total):
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            rmse = float(torch.sqrt(loss.detach()))
            run.metric_sink.log(epoch, {self.metric_name(): rmse})
            if epoch % self._ckpt_every == 0:
                self._checkpoint(run, model, operator, size, epoch, opt, total, rmse)
            if self._epoch_delay:
                time.sleep(self._epoch_delay)
        self._checkpoint(run, model, operator, size, total - 1, opt, total, rmse)

    def predict(self, checkpoint: str, x: Any) -> Any:
        """Restore a field from a degraded input `x` (a (1,1,H,W) or (H,W) tensor)."""
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = _build(ckpt["operator"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        t = torch.as_tensor(x, dtype=torch.float)
        while t.dim() < 4:
            t = t.unsqueeze(0)
        with torch.no_grad():
            return model(t)[0, 0]

    def viewer(self) -> Viewer:
        from heads.weather.viewer import WeatherViewer

        return WeatherViewer()

    # --- internals -------------------------------------------------------

    def _checkpoint(
        self,
        run: Run,
        model: nn.Module,
        operator: str,
        size: int,
        epoch: int,
        opt: torch.optim.Optimizer,
        total: int,
        rmse: float,
    ) -> None:
        Path(run.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "operator": operator,
                "size": size,
                "epoch": epoch,
                "optimizer": opt.state_dict(),
            },
            Path(run.checkpoint_dir) / "model.pt",
        )
        (Path(run.checkpoint_dir) / "meta.json").write_text(
            json.dumps(
                {
                    "operator": operator,
                    "metric": self.metric_name(),
                    "epoch": epoch,
                    "total": total,
                    "rmse": rmse,
                }
            )
        )

    def _resume(
        self,
        run: Run,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        operator: str,
        size: int,
    ) -> int:
        path = Path(run.checkpoint_dir) / "model.pt"
        if not path.exists():
            return 0
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("operator") != operator or ckpt.get("size") != size:
            return 0
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        return int(ckpt.get("epoch", 0)) + 1


# The registry loads this attribute (see spine/registry.py).
HEAD = WeatherHead()

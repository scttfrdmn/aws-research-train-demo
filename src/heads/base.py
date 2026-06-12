"""The domain-head contract — the central artifact (SPEC §2.1).

Every head implements this one interface. It is the seam that makes the spine
domain-agnostic: the spine, the scripts, the board, and the viewers talk to this
contract and never to a concrete head. If you find yourself wanting to branch on
`domain == ...` anywhere in `src/spine/`, the contract is missing a method — add
it here instead (SPEC §8).

Changing this interface in a breaking way is a MAJOR SemVer event even pre-1.0
(CLAUDE.md). Treat it as load-bearing.

A head is declared as a `Protocol` (structural): a head just has to *have* these
members with these shapes — it does not have to inherit anything. The
conformance test (`tests/test_contract.py`) checks every registered head against
this protocol at runtime, which is the guard rail that stops a new head from
special-casing the spine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from spine.run import Run


@runtime_checkable
class SweepAxis(Protocol):
    """One scientific dimension to sweep, plus a suggested grid of values.

    Heads return a list of these from `sweep_axes()`. The board renders the
    chosen point of the grid as the hypothesis chip via `tile_label(hp)`; the
    values themselves drive `scripts/sweep.py`'s fan-out.
    """

    name: str
    """The hyperparameter name (also the `--<name>` CLI flag, e.g. `feat`)."""

    values: list[str]
    """The suggested grid for this axis (e.g. ['ecfp', 'graph', 'graph+3d'])."""


@runtime_checkable
class Viewer(Protocol):
    """What the result viewers (marimo + Streamlit) render for this head.

    Kept abstract on purpose: the viewers stay generic and dispatch to whatever
    the head declares here, so every visual surface is domain-blind (SPEC §5).
    Concrete shape is settled when the first viewer lands (v0.3); for now a head
    returns an object carrying the render callables the viewers invoke.
    """

    def render(self, checkpoint: str, x: Any) -> Any:
        """Produce the head-native view of one input under one checkpoint.

        e.g. molecular: a molecule render + predicted-vs-actual scatter.
        """
        ...


@runtime_checkable
class Head(Protocol):
    """The interface every domain head implements (SPEC §2.1).

    Heads train their own way (HF Trainer, PyG, neuralop, a custom loop) but
    report through one `MetricSink` and checkpoint to one `Run` contract. That
    uniformity is what makes the board, the sweep, and the compare view
    domain-blind.
    """

    name: str
    """Identifier used by `--domain` and written to the `Domain` job tag."""

    dependency_group: str
    """The uv dependency group this head's deps live in (kept out of core)."""

    def prepare_data(self, data_dir: str, split: str) -> str:
        """Stage and return the dataset path for `split` under `data_dir`.

        Must provide a tiny local sample for smoke mode and an S3-backed path
        for cloud runs. Returns the resolved path the head reads from in `fit`.
        """
        ...

    def fit(self, run: Run, hp: dict[str, Any]) -> None:
        """Train, however this domain wants.

        `run` carries `checkpoint_dir`, `metric_sink`, `data_dir`, and
        `max_steps` (set for smoke mode). The head reports progress via
        `run.metric_sink.log(step, {...})` and writes checkpoints under
        `run.checkpoint_dir`. `hp` is the parsed hyperparameter dict.
        """
        ...

    def predict(self, checkpoint: str, x: Any) -> Any:
        """Load `checkpoint` and run inference on one input `x`."""
        ...

    def sweep_axes(self) -> list[SweepAxis]:
        """Declare the scientific sweep dimensions and a suggested grid.

        Science heads sweep featurization/architecture/etc. The LLM head sweeps
        lr × rank — honestly a CS knob; keep the contrast, don't hide it.
        """
        ...

    def tile_label(self, hp: dict[str, Any]) -> str:
        """The short hypothesis chip the board shows (→ `Hypothesis` tag).

        Must fit in a SageMaker tag value (≤256 chars) — chips, not sentences
        (SPEC §9.4). e.g. 'feat=graph · arch=deep'.
        """
        ...

    def metric_name(self) -> str:
        """The eval metric the board/compare views key on (→ `Metric` tag).

        The bare string (e.g. 'rmse', 'eval_loss') equals the CloudWatch metric
        Name and the estimator's `metric_definitions[].Name` (SPEC §9.2).
        """
        ...

    def viewer(self) -> Viewer:
        """Declare what the result viewers render for this head (SPEC §5)."""
        ...

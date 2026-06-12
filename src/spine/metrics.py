"""MetricSink — the uniform reporting seam (SPEC §2.2).

A head calls `sink.log(step, {...})` and the spine fans that out to every
registered destination: stdout (always, so SageMaker's log scraper can apply the
`metric_definitions` regex → CloudWatch), and — once it lands — the board feed.
Uniform reporting is what lets the board read every head identically; the head
never knows how many destinations there are.
"""

from __future__ import annotations

from typing import Protocol


class MetricDestination(Protocol):
    """One place metrics go. The sink fans out to a list of these."""

    def emit(self, step: int, metrics: dict[str, float]) -> None: ...


class StdoutDestination:
    """Print one `key=value` line per metric, prefixed so a `metric_definitions`
    regex can scrape it into CloudWatch. Always present.
    """

    def emit(self, step: int, metrics: dict[str, float]) -> None:
        parts = " ".join(f"{k}={v:.6g}" for k, v in metrics.items())
        print(f"[metric] step={step} {parts}", flush=True)


class MetricSink:
    """Fan-out sink handed to a head via the `Run` context.

    Stage 2 (smoke) needs only stdout. Cloud destinations (board feed, explicit
    CloudWatch puts) are appended as they land; the head-facing `log` API does
    not change.
    """

    def __init__(self, destinations: list[MetricDestination] | None = None) -> None:
        self._destinations: list[MetricDestination] = destinations or [
            StdoutDestination()
        ]

    def add(self, destination: MetricDestination) -> None:
        """Register another destination (e.g. the board feed at stage 4)."""
        self._destinations.append(destination)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Report metrics for `step` to every destination."""
        for destination in self._destinations:
            destination.emit(step, metrics)

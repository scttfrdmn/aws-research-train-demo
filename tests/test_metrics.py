"""MetricSink fan-out test (CLAUDE.md: test the sink fan-out, not the cloud SDK)."""

from __future__ import annotations

from spine.metrics import MetricSink


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, float]]] = []

    def emit(self, step: int, metrics: dict[str, float]) -> None:
        self.calls.append((step, dict(metrics)))


def test_log_fans_out_to_every_destination() -> None:
    a, b = _Recorder(), _Recorder()
    sink = MetricSink(destinations=[a, b])
    sink.log(3, {"rmse": 0.41})
    assert a.calls == b.calls == [(3, {"rmse": 0.41})]


def test_add_registers_another_destination() -> None:
    a = _Recorder()
    sink = MetricSink(destinations=[a])
    late = _Recorder()
    sink.add(late)
    sink.log(1, {"loss": 2.0})
    assert a.calls == [(1, {"loss": 2.0})]
    assert late.calls == [(1, {"loss": 2.0})]


def test_default_sink_has_a_destination() -> None:
    # default (stdout) sink must not crash when logging
    MetricSink().log(0, {"x": 1.0})

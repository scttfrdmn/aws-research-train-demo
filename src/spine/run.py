"""The Run context handed to a head's `fit()` (SPEC §2.2).

Carries everything a head needs to train and report without knowing whether it
runs on a laptop or in a managed cloud job: where to write checkpoints, where
its data is, the metric sink to report through, and `max_steps` (set in smoke
mode to cut a real forward/backward to seconds).
"""

from __future__ import annotations

from dataclasses import dataclass

from spine.metrics import MetricSink


@dataclass(frozen=True)
class Run:
    """Immutable per-invocation context.

    The same object is produced locally (stage 2) and in the cloud (stage 3+);
    only the paths differ (local dir vs. S3-backed mount). A head reads these
    and never branches on where it is running.
    """

    checkpoint_dir: str
    """Where the head writes/reads checkpoints. Local path or the container's
    `checkpoint_local_path` that SageMaker syncs to `checkpoint_s3_uri`."""

    data_dir: str
    """Resolved dataset root (from `head.prepare_data`)."""

    metric_sink: MetricSink
    """Uniform reporting seam; head calls `metric_sink.log(step, {...})`."""

    max_steps: int | None = None
    """Smoke cap (`--max-steps`). None = train to the head's own stopping rule."""

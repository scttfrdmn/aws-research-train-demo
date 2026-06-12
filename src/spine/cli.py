"""CLI — arg parsing and `--domain` dispatch (SPEC §2.2).

`train.py` is the thin entry; the real wiring lives here. Hyperparameters are
CLI arguments, never module constants (CLAUDE.md): the spine parses the
domain-neutral flags it owns (`--domain`, `--max-steps`, dirs) and passes the
rest through to the head as its `hp` dict, so adding a head's scientific knob
never touches the spine.

The dispatch must stay domain-blind: no `if domain == ...`. The registry resolves
the name to a head; the spine only talks to the contract.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from typing import Any

from spine import registry
from spine.metrics import MetricSink
from spine.run import Run


def _parse_known(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Parse the spine-owned flags; return the rest for the head's `hp`."""
    parser = argparse.ArgumentParser(prog="train.py", add_help=True)
    parser.add_argument(
        "--domain", required=True, help=f"head to run; one of {registry.names()}"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="smoke cap: run this many steps for a fast real fwd/bwd",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("SM_CHANNEL_TRAINING", None),
        help="dataset root; defaults to SageMaker's training channel if set",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"),
        help="where the head writes checkpoints (synced to S3 in cloud runs)",
    )
    return parser.parse_known_args(argv)


def _hp_from_extras(extras: list[str]) -> dict[str, Any]:
    """Turn passthrough `--key value` / `--flag` tokens into the head's hp dict.

    The spine does not know a head's hyperparameter names, so it does not
    validate them — it forwards them. The head owns their meaning.
    """
    hp: dict[str, Any] = {}
    i = 0
    while i < len(extras):
        token = extras[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token[2:].replace("-", "_")
        if i + 1 < len(extras) and not extras[i + 1].startswith("--"):
            hp[key] = extras[i + 1]
            i += 2
        else:
            hp[key] = True  # bare flag
            i += 1
    return hp


def main(argv: list[str] | None = None) -> None:
    """Entry point: resolve the head, build the Run, dispatch to `fit`."""
    args, extras = _parse_known(argv)
    head = registry.load(args.domain)
    hp = _hp_from_extras(extras)

    data_dir = args.data_dir or head.prepare_data(
        tempfile.mkdtemp(prefix=f"{head.name}-data-"), split="train"
    )
    run = Run(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=data_dir,
        metric_sink=MetricSink(),
        max_steps=args.max_steps,
    )
    head.fit(run, hp)

"""Stage 3 — submit ONE parameterized SageMaker training job (SPEC §4, issue #1).

Same `train.py` the laptop smoke-tested; S3 data + checkpoints; the §9 job tags
the Go board reads. This is the Python side of the head→board seam: it *can*
import the head (unlike the board), so it bakes `head.tile_label(hp)` /
`head.metric_name()` / `head.name` into tags at submit time.

Written against the **installed** sagemaker 3.13.1 API (`ModelTrainer`; the
classic `PyTorch` estimator is gone — see issue #1 code-time correction), not
the report's classic-estimator assumption.

SAFETY: defaults to `--dry-run` (build + print the plan, no AWS spend). A real
job needs `--submit`. Read-only/no-op without it.

    uv run --group cloud --group molecular python scripts/submit.py \
        --domain molecular --feat graph --depth deep \
        --instance ml.g5.xlarge --sweep mol-esol-20260612-a --seq 1 \
        --s3-bucket my-bucket --role-arn arn:aws:iam::...:role/SageMakerRole \
        --submit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from heads.base import Head  # noqa: E402
from spine import registry  # noqa: E402

# The head's training loop prints `[metric] step=N rmse=0.41` (see
# spine/metrics.py StdoutDestination). SageMaker scrapes stdout with this regex
# into CloudWatch under the metric's bare name (SPEC §9.2). One capture group.
_METRIC_REGEX = r"{name}=([0-9.eE+-]+)"


def _job_name(sweep: str, seq: int) -> str:
    """`<Sweep>-NN` — the name prefix the board scopes a sweep by (SPEC §9.3)."""
    return f"{sweep}-{seq:02d}"


def _tags(head: Head, hp: dict[str, Any], instance: str, spot: bool) -> dict[str, str]:
    """The §9 job tags. Direction suffix (↓ lower-is-better) on Metric."""
    return {
        "Sweep": hp["_sweep"],
        "Hypothesis": head.tile_label(hp),
        "Metric": f"{head.metric_name()}↓",  # all current heads minimize
        "Domain": head.name,
        "Instance": instance,
        "Spot": "true" if spot else "false",
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve the head and assemble everything the job needs — no AWS calls."""
    head = registry.load(args.domain)
    hp = {
        "feat": args.feat,
        "depth": args.depth,
        "epochs": args.epochs,
        "_sweep": args.sweep,
    }
    job_name = _job_name(args.sweep, args.seq)
    tags = _tags(head, hp, args.instance, args.spot)
    metric = head.metric_name()
    return {
        "job_name": job_name,
        "head": head,
        "hp": {k: v for k, v in hp.items() if not k.startswith("_")},
        "tags": tags,
        "metric_definitions": [
            {"name": metric, "regex": _METRIC_REGEX.format(name=metric)}
        ],
        "instance": args.instance,
        "spot": args.spot,
        "s3_data": f"s3://{args.s3_bucket}/{args.domain}/data/"
        if args.s3_bucket
        else None,
        "checkpoint_s3": f"s3://{args.s3_bucket}/{args.sweep}/{job_name}/checkpoints/"
        if args.s3_bucket
        else None,
    }


def submit(plan: dict[str, Any], args: argparse.Namespace) -> str:
    """Build a ModelTrainer from the plan and call train(). Spends money."""
    from sagemaker.core.image_uris import retrieve
    from sagemaker.core.shapes.shapes import (
        CheckpointConfig,
        MetricDefinition,
        StoppingCondition,
        Tag,
    )
    from sagemaker.core.training.configs import Compute, InputData, SourceCode
    from sagemaker.train.model_trainer import ModelTrainer

    image = args.image or retrieve(
        framework="pytorch",
        region=args.region,
        version=args.framework_version,
        py_version=args.py_version,
        instance_type=args.instance,
        image_scope="training",
    )

    trainer = ModelTrainer(
        training_image=image,
        role=args.role_arn,
        base_job_name=plan["job_name"],
        source_code=SourceCode(source_dir=str(REPO), entry_script="train.py"),
        compute=Compute(
            instance_type=args.instance,
            instance_count=1,
            enable_managed_spot_training=args.spot,
        ),
        stopping_condition=StoppingCondition(
            max_runtime_in_seconds=args.max_runtime,
            max_wait_time_in_seconds=args.max_runtime if args.spot else None,
        ),
        checkpoint_config=CheckpointConfig(s3_uri=plan["checkpoint_s3"]),
        hyperparameters={**plan["hp"], "domain": args.domain},
        tags=[Tag(key=k, value=v) for k, v in plan["tags"].items()],
    ).with_metric_definitions(
        [
            MetricDefinition(name=m["name"], regex=m["regex"])
            for m in plan["metric_definitions"]
        ]
    )

    inputs = [InputData(channel_name="training", data_source=plan["s3_data"])]
    trainer.train(input_data_config=inputs, wait=args.wait, logs=args.wait)
    job_name: str = plan["job_name"]
    return job_name


def _print_plan(plan: dict[str, Any]) -> None:
    print("=== submit plan (dry-run; pass --submit to run) ===")
    print(f"  job name        : {plan['job_name']}")
    print(f"  instance / spot : {plan['instance']} / {plan['spot']}")
    print(f"  s3 data         : {plan['s3_data']}")
    print(f"  checkpoint s3   : {plan['checkpoint_s3']}")
    print(f"  hyperparameters : {plan['hp']}")
    print("  tags (SPEC §9):")
    for k, v in plan["tags"].items():
        print(f"      {k} = {v}")
    print(f"  metric defs     : {plan['metric_definitions']}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="submit.py")
    p.add_argument("--domain", required=True)
    p.add_argument("--feat", default="ecfp")
    p.add_argument("--depth", default="shallow")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--instance", default="ml.g5.xlarge")
    p.add_argument("--spot", action="store_true")
    p.add_argument("--sweep", required=True, help="sweep id == job-name prefix (§9.3)")
    p.add_argument("--seq", type=int, default=1, help="job sequence within the sweep")
    p.add_argument("--s3-bucket", default=None)
    p.add_argument("--role-arn", default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--image", default=None, help="override DLC image URI")
    p.add_argument("--framework-version", default="2.10")
    p.add_argument("--py-version", default="py313")
    p.add_argument("--max-runtime", type=int, default=3600)
    p.add_argument("--wait", action="store_true", help="stream logs until done")
    p.add_argument(
        "--submit", action="store_true", help="actually submit (spends money)"
    )
    args = p.parse_args(argv)

    plan = build_plan(args)
    _print_plan(plan)

    if not args.submit:
        print("\n[dry-run] not submitting. Re-run with --submit to launch.")
        return
    for need in ("s3_bucket", "role_arn", "region"):
        if getattr(args, need) is None:
            p.error(f"--{need.replace('_', '-')} is required with --submit")
    name = submit(plan, args)
    print(f"\n[submitted] training job: {name}")


if __name__ == "__main__":
    main()

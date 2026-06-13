"""Stage 4 — fan out a sweep across the head's scientific axes (SPEC §4, §2.2).

Reads `head.sweep_axes()`, builds the cartesian grid (or a caller-narrowed
subset), and submits one job per point — each tagged `Sweep=<id>` with the
`<Sweep>-NN` name so the board scopes them together (§9.3). The grid of live
curves is the argument that this is research.

This is the spot where the head→board coupling is written: `sweep.py` calls
`head.tile_label(hp)` / `head.metric_name()` (Python may import the head) and
bakes them into tags; the Go board reads the tags back. Neither imports the
other. Reuses `scripts/submit.py` so the per-job plan/submit path is identical.

SAFETY: dry-run by default — prints the fan-out plan, no AWS spend. `--submit`
launches the whole grid (N jobs). Confirm account/region before using it.

    uv run --group cloud --group molecular python scripts/sweep.py \
        --domain molecular --sweep mol-esol-20260612-a \
        --axes feat,depth --s3-bucket b --role-arn arn:... --region us-west-2 \
        --submit
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import submit as submit_mod  # noqa: E402
from heads.base import Head  # noqa: E402
from spine import registry  # noqa: E402


def _grid(
    head: Head, axes: list[str] | None, limit: int | None
) -> list[dict[str, Any]]:
    """Cartesian product of the requested axes' suggested values.

    `axes=None` uses every axis the head declares; otherwise only the named
    ones (the rest fall back to their first value via submit.py defaults).
    """
    declared = {a.name: a.values for a in head.sweep_axes()}
    chosen = axes or list(declared)
    unknown = [a for a in chosen if a not in declared]
    if unknown:
        raise SystemExit(
            f"unknown sweep axes {unknown}; head declares {list(declared)}"
        )
    names = chosen
    combos = list(itertools.product(*(declared[n] for n in names)))
    points = [dict(zip(names, combo, strict=True)) for combo in combos]
    return points[:limit] if limit else points


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="sweep.py")
    p.add_argument("--domain", required=True)
    p.add_argument("--sweep", required=True, help="sweep id == job-name prefix (§9.3)")
    p.add_argument("--axes", default=None, help="comma list; default = all head axes")
    p.add_argument("--limit", type=int, default=None, help="cap the grid size")
    # CPU default — see submit.py: g5 quota is 1, CPU quota 20-30; a parallel
    # sweep needs CPU and the ESOL models are tiny.
    p.add_argument("--instance", default="ml.c5.xlarge")
    p.add_argument("--spot", action="store_true")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--s3-bucket", default=None)
    p.add_argument("--role-arn", default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--framework-version", default="2.8")
    p.add_argument("--py-version", default="py312")
    p.add_argument("--max-runtime", type=int, default=3600)
    p.add_argument(
        "--submit", action="store_true", help="launch the grid (spends money)"
    )
    args = p.parse_args(argv)

    head = registry.load(args.domain)
    axes = args.axes.split(",") if args.axes else None
    points = _grid(head, axes, args.limit)

    print(f"=== sweep {args.sweep}: {len(points)} jobs across {axes or 'all axes'} ===")
    submitted = []
    for seq, point in enumerate(points, start=1):
        job_args = argparse.Namespace(
            domain=args.domain,
            feat=point.get("feat", "ecfp"),
            depth=point.get("depth", "shallow"),
            epochs=args.epochs,
            instance=args.instance,
            spot=args.spot,
            sweep=args.sweep,
            seq=seq,
            s3_bucket=args.s3_bucket,
            role_arn=args.role_arn,
            region=args.region,
            image=args.image,
            framework_version=args.framework_version,
            py_version=args.py_version,
            max_runtime=args.max_runtime,
            wait=False,
        )
        plan = submit_mod.build_plan(job_args)
        print(f"\n--- job {seq}/{len(points)} :: {plan['tags']['Hypothesis']} ---")
        submit_mod._print_plan(plan)
        if args.submit:
            for need in ("s3_bucket", "role_arn", "region"):
                if getattr(job_args, need) is None:
                    p.error(f"--{need.replace('_', '-')} is required with --submit")
            submitted.append(submit_mod.submit(plan, job_args))

    if not args.submit:
        print(
            f"\n[dry-run] {len(points)} jobs planned; re-run with --submit to launch."
        )
    else:
        print(f"\n[submitted] {len(submitted)} jobs: {submitted}")


if __name__ == "__main__":
    main()

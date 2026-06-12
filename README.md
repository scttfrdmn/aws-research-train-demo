# aws-research-train-demo

A SageMaker training demo that shows how research training **actually** happens —
authored and validated locally, executed in the cloud, fanned out into a sweep,
watched live on a board, compared across runs, and only *then* viewed. The
notebook is the last few feet, not the engine.

One **neutral spine** runs any of four **domain heads** behind a common contract
(`src/heads/base.py`). The **molecular** head is built first, end to end;
weather / genomics / llm come later against the unchanged contract.

> **What & why:** [`SPEC.md`](SPEC.md). **How code is written:**
> [`CLAUDE.md`](CLAUDE.md). **Plans/status:** GitHub issues — not this repo.

## Quickstart

```bash
uv sync                      # core spine only
uv sync --group molecular    # + the molecular head's deps (RDKit, PyG)
uv run python train.py --domain molecular --feat graph --depth deep --max-steps 5
```

## The seven-stage arc

Each stage produces a **file or a job** whose identity outlives the session, and
each inverts the tell that training-in-a-notebook is how research is done.

1. **Author** — `train.py --domain <h>`, hyperparameters as args. *Inverts:*
   notebook code as throwaway kernel state.
2. **Smoke-test locally** — `--max-steps 5`, a real fwd/bwd in seconds.
   *Inverts:* paying the cloud to find a typo.
3. **Submit one cloud job** — same file, S3 data + checkpoints.
   *Inverts:* the laptop pretending to be the cluster.
4. **Fan out a sweep** — 4–6 jobs across the head's **scientific** axis, live on
   the board. *Inverts:* one kernel, one run.
5. **Spot + checkpoint** — survive a reclaim, resume from S3.
   *Inverts:* the canned demo where nothing goes wrong.
6. **Compare runs** — overlay/sort by the head's metric, in marimo.
   *Inverts:* nothing — this stage simply *is* research, made visible.
7. **Two viewers** — marimo (operator, reactive) + Streamlit (collaborator,
   drop-in-your-input). *Inverts:* the notebook as the engine.

## Three surfaces

| surface | tech | audience |
|---|---|---|
| **Board** | Go (`dashboard/`, `aws-sdk-go-v2`, read-only) | the room, live during training |
| **marimo** (`analysis/explore.py`) | Python, reactive | the operator |
| **Streamlit** (`app/`) | Python, frozen | collaborators |

## Layout

See [`SPEC.md` §6](SPEC.md). The spine is in `src/spine/`, heads in
`src/heads/<name>/`, the contract in `src/heads/base.py`, the Go board in
`dashboard/`.

## Status

Tracked in [GitHub issues](https://github.com/scttfrdmn/aws-research-train-demo/issues)
and milestones — never in this repo.

# aws-research-train-demo — Specification (v2)

> Renamed from `aws-llm-finetune-demo`: the LLM is now one head among four, not
> the headliner.

A SageMaker training demo that shows how research training **actually** happens —
authored and validated locally, executed in the cloud, fanned out into a sweep,
watched live on a board, compared across runs, and only *then* viewed. The
notebook is the last few feet, not the engine.

One **neutral spine** runs any of four **domain heads** behind a common contract.
Build the molecular head first; the architecture must run all four.

This spec is the source of truth for **what** and **why**. Coding conventions are
in `CLAUDE.md`. Task tracking lives in GitHub — not here.

---

## 1. Thesis (unchanged)

The demo refutes one error: that training a model inside a Jupyter notebook is how
research is done. It isn't. Training is a batch job; research iteration is
iteration *across runs*, not *within* one. A single clean notebook training cell
is a tell that the author has never run the fortieth job.

Every stage is the **inverse of that tell**, and every stage produces a **file or
a job** — an artifact whose identity outlives the session. Nothing important lives
in kernel state.

Two audiences, satisfied by the same spine:

- **AWS SAs / sales:** SageMaker visibly does the heavy lifting the laptop can't
  (managed jobs, spot, checkpointing, tracking), shown on a live board.
- **Academic researchers / PIs:** nothing is locked in. The same code runs locally
  and in the cloud; data gravity never moves into a proprietary surface. SageMaker
  is the **executor**, not the **owner**.

Closing line for both rooms: *the laptop authored and validated it, the cloud
executed it, the sweep is where the science was, and the notebook only showed up
at the end as a viewer.*

---

## 2. Architecture: neutral spine + domain heads

The spine owns the **lifecycle**; a head owns the **science**. The spine never
special-cases a head — it only talks to the contract.

### 2.1 The domain-head contract (`src/heads/base.py`) — the central artifact

Every head implements one interface. This is the seam that makes the spine
domain-agnostic; getting it right is the whole job.

A head must provide:

- **`name`** — identifier used by `--domain`.
- **`dependency_group`** — the uv group its deps live in (kept out of core).
- **`prepare_data(data_dir, split)`** — stage/return the dataset. Must provide a
  tiny local sample for smoke mode and an S3-backed path for cloud runs.
- **`fit(run, hp)`** — train however this domain wants (HF Trainer, PyG, neuralop,
  custom loop). Receives a `Run` context from the spine carrying
  `checkpoint_dir`, `metric_sink`, `data_dir`, and `max_steps` (smoke). Trains its
  own way; reports through the sink.
- **`predict(checkpoint, x)`** — load a checkpoint, run inference on one input.
- **`sweep_axes()`** — declares the **scientific** dimensions to sweep and a
  suggested grid. (Science heads: featurization/architecture/etc. LLM head:
  `lr × rank` — honestly a CS knob; keep the contrast, don't hide it.)
- **`tile_label(hp)`** — the short hypothesis chip the board shows for a run
  (e.g. `feat=graph · arch=deep`).
- **`metric_name()`** — the eval metric the board/compare views key on
  (RMSE, eval_loss, etc.).
- **`viewer()`** — declares what the result viewer renders (see §5).

### 2.2 Spine responsibilities

- `src/spine/cli.py` — arg parsing, `--domain` dispatch, `--max-steps` smoke flag.
- `src/spine/run.py` — the `Run` context handed to `fit()`.
- `src/spine/metrics.py` — `MetricSink`: head calls `sink.log(step, {...})`;
  the spine fans it to CloudWatch metrics **and** the board. Uniform reporting is
  what lets the board read every head identically.
- `src/spine/registry.py` — discovers heads by `name`, loads only the requested
  one (so installing `molecular` doesn't drag in `neuralop`).

**Job metadata is the head→board bridge.** `scripts/sweep.py` (Python) calls
`head.tile_label(hp)` and `head.metric_name()` and writes them into each job's
**tags** (`Hypothesis=…`, `Metric=…`, `Sweep=<id>`) plus the metric definition at
submit time. The Go board reads those tags — it never imports head code. This is
what lets the board be Go while the heads stay Python (see §5.1).

The key seam: **heads train their own way, but report through one sink and
checkpoint to one contract.** That uniformity is what makes the board, the sweep,
and the compare view domain-blind.

---

## 3. The four heads

Build order: **molecular first, end to end**, then add the others against the same
contract.

| head | task | sweep axis (the hypothesis) | metric | viewer |
|------|------|------------------------------|--------|--------|
| **molecular** *(build first)* | SMILES → property regression (MoleculeNet: ESOL/Tox21) | featurization {ecfp, graph, graph+3d} × depth {shallow, deep} | RMSE | molecule render + predicted-vs-actual scatter |
| **weather** *(showcase)* | spatial field predict (downscaling / nowcasting; FNO or U-Net) | operator {fno, unet} × lead-time / resolution | field RMSE | predicted | truth | error triptych (heatmaps) |
| **genomics** *(wedge)* | DNA → regulatory signal (small DeepSEA/Basset); pair with phantom-science synthetic data for zero license friction | receptive-field {small, large} × arch {cnn, dilated} | auROC / corr | sequence saliency + motif logo |
| **llm** *(one head among four)* | small open-weight instruct LoRA fine-tune | **lr × rank** (CS knob — keep the contrast) | eval_loss | base output vs tuned output, side by side |

The molecular head is the reference because it's the lowest-risk path to a thing
that runs: tiny models, minutes to train, license-clean well-trodden benchmarks,
gorgeous and legible output. Weather is the showcase (most arresting visual; it's
the real UChicago engagement; FNO ties to the trnsci / post-FP64 thesis). Genomics
is the wedge (Queryabl BAMQ/VCFQ; synthetic data sidesteps licensing). LLM proves
the spine is genuinely domain-blind.

**Behavior-delta requirement (per head):** the trained-vs-untrained difference
must be visible to the naked eye in the Streamlit viewer. If it isn't, the
dataset/task is wrong.

---

## 4. The arc + the visual spine

Seven stages; each names the tell it inverts. The **board** is what makes stages
3–6 watchable by a non-expert. The board reads job state from the control plane
(`DescribeTrainingJob` + CloudWatch metrics) and renders runs as **objects with
state** — never cells. Every visual is a window onto infrastructure; no visual is
a chart drawn inside the training notebook.

1. **Author in VSCode against the repo.** `train.py --domain <h>` with
   hyperparameters as args; nothing hardcoded. *Inverts:* notebook code is kernel
   state that doesn't survive leaving the room.
2. **Smoke-test locally.** `--max-steps 5` on the home-lab GPU; optional SageMaker
   **local mode**. Board tile flips `PENDING → ✓` in seconds. *Inverts:* paying
   cloud to find a typo.
3. **Submit one cloud job, parameterized.** Same file, S3 data, S3 checkpoints.
   Board tile walks the state machine `PENDING → IN PROGRESS → COMPLETED`.
   *Inverts:* the laptop pretending to be the cluster.
4. **Fan out into a sweep.** 4–6 jobs at once across the head's **scientific**
   sweep axis. Board lights up six tiles, curves drawing live in parallel.
   *Inverts:* one kernel, one run. **The grid of live curves is the argument** that
   this is research.
5. **Spot + checkpoint, surviving interruption.** Managed spot +
   `checkpoint_s3_uri`. Board tile goes red `reclaim` → amber `RESUMING ⟲` →
   rejoins green. *Inverts:* the canned demo's "nothing goes wrong."
6. **Compare the runs.** Curves overlay / sort by `metric_name()`, in **marimo**
   (show Experiments once if desired, but compare in the portable surface).
   *Inverts:* nothing — this stage simply *is* research, made visible.
7. **Land in two viewers, by audience.**
   - **marimo (operator):** reactive `.py`. Swap the checkpoint path at the top;
     every downstream cell re-derives via `head.viewer()`. *Credible reason:* you
     refuse to decide on a number that no longer corresponds to any runnable state
     — the "verifiable" brand, dogfooded; the defensible answer to "why not
     Jupyter."
   - **Streamlit (collaborator):** frozen interface. A scientist drops in *their
     own* input (a SMILES string, a weather patch, a sequence, a prompt) and sees
     base-vs-trained. The endpoints prove taste, not skill — the proof was banked
     upstream.

---

## 5. Viewers are domain-native, driven by the head

`head.viewer()` declares what the result viewers render, so marimo and Streamlit
stay generic and dispatch to the head. This keeps every visual surface
domain-blind, mirroring the spine.

The board is a special case: it's **Go**, so it can't call Python head methods.
It reads what it needs from **job metadata** instead (see §5.1) — which keeps it
fully decoupled from the head code.

### 5.1 Three surfaces, three audiences, three tech choices (on purpose)

| surface | tech | audience | served by |
|---|---|---|---|
| **Board** | **Go** (`aws-sdk-go-v2`) | the room, live during training | `dashboard/` binary, e.g. :8080 |
| **marimo** | marimo (`.py`) | the operator — reactive analysis cockpit | `marimo run analysis/explore.py` (read-only app) |
| **Streamlit viewer** | Streamlit | collaborators/scientists — drop-in-your-input | `app/`, e.g. :8501 |

**Principle:** Go for control-plane / infra clients, Python only where the ML
ecosystem forces it. The board reads job state — warranted Go. The spine and heads
are Python because PyTorch/PEFT/RDKit force it. Cleanly partitioned by directory.

**The board (Go).** Serves the approved `board_template.html` at `/` plus a JSON
feed at `/api/jobs`. The feed does read-only AWS calls — `ListTrainingJobs`
(scoped to the current sweep by tag), `DescribeTrainingJob`, CloudWatch
`GetMetricData`. The page's JS fetches `/api/jobs` every 3–5s (relative URL) and
mutates **only** the tiles/curves — smooth, no flicker. Compiles to one static
binary (tiny Fargate image; runs anywhere). Read-only IAM:
`sagemaker:ListTrainingJobs`, `sagemaker:DescribeTrainingJob`,
`cloudwatch:GetMetricData`.

**How the Go board stays decoupled from the Python heads.** The board never
imports head code. Instead, `sweep.py` (Python — it *can* call the head) bakes the
head-derived values into each job at submit time:

- `head.tile_label(hp)` → a job **tag** (e.g. `Hypothesis=feat=graph · arch=deep`)
- `head.metric_name()` → a job **tag** + the job's metric definition (e.g. `Metric=rmse`)
- a `Sweep=<sweep-id>` tag so the board can scope `ListTrainingJobs` to this sweep

The board reads those tags off `DescribeTrainingJob` and pulls the named series
from CloudWatch. **The head→board coupling goes through SageMaker job metadata, not
a shared import** — which is exactly what lets the board be Go while the heads stay
Python.

**The result viewers (marimo + Streamlit)** are Python and *do* dispatch to
`head.viewer()`. marimo is the operator's reactive cockpit (`marimo edit` to
author, `marimo run` to serve read-only); Streamlit is the collaborator-facing
drop-in-your-input viewer.

### 5.2 Serving through the Studio proxy — all three at once

You can proxy **several apps simultaneously**, each on its own port:
jupyter-server-proxy exposes any localhost port at `<base>/proxy/<port>/`, and this
works for all listening ports at once. Board (:8080), `marimo run` (its port),
and Streamlit viewer (:8501) coexist, each at its own proxy URL.

Gotchas (resolve in verify-first):

- The proxy is bound to the **JupyterLab** app, **not** Code Editor. Author
  wherever; run the *served* apps from JupyterLab terminals. Base URL:
  `…studio.<region>.sagemaker.aws/jupyterlab/default/proxy/<port>/`.
- **Streamlit** under a path prefix needs base-URL / XSRF config or its assets and
  websocket 404 — verify current flags (`--server.baseUrlPath`, XSRF).
- **marimo** under a path prefix has the same class of base-URL gotcha — verify its
  proxy/base-url flag. The proxied surface is `marimo run` (read-only), not
  `marimo edit`.
- The Go board must use **relative** fetch URLs because the proxy strips the
  `/proxy/<port>/` prefix.
- VPC-only mode: the security group may need to allow instance loopback.

**Deployments — same apps, differing only in where/reach/creds:**

| | run | reach | creds |
|---|---|---|---|
| **Studio** *(primary)* | start board + marimo + viewer from a JupyterLab terminal, each on its port | `…/jupyterlab/default/proxy/<port>/` per app | Studio execution role |
| **Local** | same apps, laptop | `localhost:<port>` per app | local AWS profile / SSO |
| **Fargate** | board binary + apps containerized | ALB / public endpoint | task role |

---

## 6. Repo layout

```
aws-research-train-demo/
  pyproject.toml          # uv; core deps + per-head dependency groups
  uv.lock                 # committed
  .python-version
  CLAUDE.md               # coding conventions
  SPEC.md                 # this file
  CHANGELOG.md            # Keep a Changelog 1.1.0
  README.md               # the arc as a runnable narrative
  train.py                # thin entry: python train.py --domain molecular ...
  src/
    spine/
      cli.py  run.py  metrics.py  registry.py
    heads/
      base.py             # THE contract
      molecular/  weather/  genomics/  llm/   # head.py · data.py · viewer.py
  scripts/
    smoke.sh              # local validate, any head (stage 2)
    submit.py             # one cloud job, --domain (stage 3)
    sweep.py              # fan-out across head.sweep_axes() (stage 4)
  dashboard/              # Go module (own go.mod) — control-plane client
    main.go               # serves board_template.html + /api/jobs (aws-sdk-go-v2, read-only)
    board_template.html   # the approved view — mission_control_mockup.html, parameterized
  analysis/
    explore.py            # marimo — reactive; `marimo edit` to author, `marimo run` to serve
  app/
    app.py                # streamlit — base-vs-trained per head (collaborator viewer)
```

The notebook is one file in `analysis/`, deliberately not the entry point. The
board is a real app in `dashboard/`, not decoration.

---

## 7. Verify-first at build time (do NOT trust memory)

Resolve against **current** docs as step zero; record findings + dates in the
relevant GitHub issue.

1. **SageMaker SDK / DLC / job API** — current estimator path, DLC image URIs, and
   any names moved by the **"SageMaker AI"** rebrand. The arc shape is stable; the
   labels are not.
2. **Per-head data + model deps** — molecular (RDKit + a GNN lib, e.g.
   torch_geometric; MoleculeNet access), weather (a small FNO lib, e.g. neuralop;
   ERA5-ish sample), genomics (phantom-science synthetic data path; motif tooling),
   llm (current HF/PEFT path; license-clean small base + visible-delta dataset).
   Each head's deps go in its **own uv group**.
3. **marimo-in-Studio serving** — confirm the Studio Jupyter-proxy port path works
   for `marimo run` as it does for Streamlit (`--server.port` → URL with `lab`
   replaced by `proxy/<port>/...`). No first-party AWS walkthrough exists for
   marimo, so budget DIY.

Honesty about the hosting ceiling: the Studio-proxy path is **dev-surface
hosting** — fine for a demo and for handing a PI a link in a working session, not a
persistent multi-user serving story (that's ECS Fargate / App Runner). Name the
ceiling before someone else does.

---

## 8. Non-goals

- **No HyperPod / multi-node distributed training.** Virtually no academic research
  needs it. The scaling axis here is *more jobs* (the sweep), not *more nodes per
  job*. Point past the demo at array-style fan-out, not NCCL across 32 nodes.
- **No training-to-completion on the laptop narrated as a cloud demo.** Local is
  for authoring and smoke-testing; the cloud executes the real run.
- **No special-casing a head in the spine.** If the spine has to branch on
  `domain ==`, the contract is wrong — fix the contract.
- **No project tracking in markdown.** Issues/milestones/labels/board in GitHub.

---

## 9. Job tag schema — the Python↔Go contract

This is the **cross-language seam** (§2.2, §5.1). The Python submitter
(`scripts/submit.py`, `scripts/sweep.py`) **writes** these values as SageMaker
training-job tags at submit time; the Go board **reads** them back. Neither side
imports the other — this schema *is* the contract. Pin it here; change it
deliberately on both sides at once.

### 9.1 How the board reads it (verify-first constraint)

Confirmed against live docs (2026-06-12): **`ListTrainingJobs` and
`DescribeTrainingJob` do NOT return tags**, and `ListTrainingJobs` has **no
tag filter**. So the board cannot scope a sweep by tag directly. The flow is:

1. `ListTrainingJobs(NameContains=<sweep-id>)` — cheap scope by **job-name
   prefix** (see §9.3), returns names + ARNs + status.
2. per job: `sagemaker:ListTags(ResourceArn)` — read the tags below.
3. live metric: CloudWatch `GetMetricData` (namespace
   `/aws/sagemaker/TrainingJobs`), or last value from
   `DescribeTrainingJob.FinalMetricDataList` (no-CloudWatch fallback).

Read-only IAM the board needs: `sagemaker:ListTrainingJobs`,
`sagemaker:DescribeTrainingJob`, `sagemaker:ListTags`,
`cloudwatch:GetMetricData`.

### 9.2 The tags

Python bakes head-derived values in via `head.tile_label(hp)`,
`head.metric_name()`, `head.name`. The board renders a tile from them — and
still **never branches on `Domain`** (it's display only).

| tag key | source | example | board use |
|---|---|---|---|
| `Sweep` | sweep id (passed to submitter) | `mol-esol-20260612-a` | group tiles; matches the name prefix (§9.3) |
| `Hypothesis` | `head.tile_label(hp)` | `feat=graph · arch=deep` | the hypothesis chip on the tile |
| `Metric` | `head.metric_name()` + direction | `rmse↓` | which series to read + sort direction (↓ = lower-is-better, ↑ = higher) |
| `Domain` | `head.name` | `molecular` | display only — board stays domain-blind |
| `Instance` | estimator `instance_type` | `ml.g5.xlarge` | the `· g5.xlarge` line on the tile |
| `Spot` | `use_spot_instances` | `true` | render the spot / reclaim affordance |

The CloudWatch metric **Name** equals the bare metric string in `Metric` (the
`↓`/`↑` suffix is stripped before lookup) and matches the estimator's
`metric_definitions[].Name` — one string, three places, by convention.

### 9.3 Job-name convention (cheap sweep scoping)

Because `ListTrainingJobs` can't filter by tag, the **sweep id is also the
job-name prefix**: jobs are named `<Sweep>-NN` (e.g. `mol-esol-20260612-a-03`).
The board scopes a sweep with `ListTrainingJobs(NameContains=<Sweep>)` *before*
the per-job `ListTags` fan-out, keeping the tag reads bounded to one sweep.

### 9.4 Constraints

- SageMaker tag **values are ≤256 chars**; `tile_label(hp)` must fit. Heads keep
  labels short (chips, not sentences).
- Tags are the **only** head→board coupling. Anything the board must render goes
  in a tag here or it doesn't reach the board — adding a board field means adding
  a tag in this section, on both sides, deliberately.

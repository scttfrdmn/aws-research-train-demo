# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

> Pre-1.0.0: the shape is still settling; a minor bump may break. The head
> contract in `src/heads/base.py` is the most load-bearing surface — a breaking
> change to it is called out loudly even pre-1.0.

## [Unreleased]

### Documentation
- `docs/cloud-setup.md` — added a **Cost & control** section: the ~20% `ml.`
  premium of SageMaker managed spot vs. self-managed EC2 spot (real Pricing/Spot
  API numbers for `c7i.large`), what the premium buys (zero-infra managed
  interrupt+resume, CloudWatch curves) and what it can't do (FIS can't target
  managed-spot instances; no 2-min countdown). Marked all demo resources torn
  down; completed the teardown runbook (EC2/FIS path + all three IAM roles).

### Changed
- **Default instance is `ml.c7i.large` (CPU), not `ml.g5.xlarge`.** This account
  caps g5 training at 1 concurrent job (CPU quota 30); a parallel sweep needs
  CPU, and the tiny ESOL models train on CPU in minutes. `c7i.large` ($0.107/hr)
  is the cheapest SageMaker training instance offered in us-west-2 — ~half of
  c5.xlarge; Graviton/AMD families aren't offered for training here. Verified: a
  6-wide CPU spot sweep ran fully parallel to completion. Override with
  `--instance`.

### Fixed
- **Board live curve: custom metrics are keyed by `TrainingJobName`**, not the
  system-metric `Host` dimension the #1 report assumed (verified via
  `list-metrics`). Sparkline now populates.
- **§9 tag values now satisfy SageMaker's charset.** `CreateTrainingJob` rejects
  tag values outside `[\p{L}\p{Z}\p{N}_.:/=+\-@]` — caught at real submit time.
  `tile_label` separator `·`→` / `; the `rmse↓` sort hint became a separate
  `MetricGoal=min` tag. Added a contract test guarding every head's tags.
- `submit.py` ships a per-head `requirements.txt` into the DLC and ignores
  `.venv/.git/data` from the source tarball; DLC default pinned to the
  installed-SDK-resolvable `pytorch 2.8 / py312` (the #1 report's 2.10/py313 is
  "Unsupported" by sagemaker 3.13.1).

### Added
- **LLM head** (`src/heads/llm/`) — small open-weight LoRA fine-tune, the fourth
  head and the proof the spine is genuinely domain-blind (SPEC §3; MINOR bump):
  it trains via the **HF `trl` SFTTrainer + PEFT** stack — a completely different
  path from the torch-loop heads — yet reports through the same `MetricSink`
  (HF `TrainerCallback` → sink) and checkpoints to the same `Run` contract. Base
  `SmolLM2-135M-Instruct` (Apache-2.0, not gated, #11); sweep axis lr × rank
  (kept honest as a CS knob); metric eval_loss; viewer = base vs tuned side by
  side. Tiny "always answer in JSON" task: base answers prose, tuned emits
  `{"answer": "..."}` — verified naked-eye delta. Smokes in ~3s on CPU (2 LoRA
  steps); saves a ~1.8 MB adapter, not the base. Spine untouched except one
  registry entry. `llm` uv group (transformers/peft/trl).
- **Genomics head** (`src/heads/genomics/`) — DNA → regulatory signal, the third
  head against the unchanged contract (SPEC §3; MINOR bump). Sweep axis
  receptive-field {small, large} × arch {cnn, dilated}; metric auROC
  (hand-rolled, torch+numpy-pure); viewer = input-gradient saliency + motif logo
  (`logomaker`). **Phantom-science synthetic data** — implant a GATA-box motif in
  positives, scrambled in negatives, generated in code (zero license friction,
  #10). Plain 1D CNN, no genomics lib. Smokes in ~2s; behavior-delta real
  (saliency mass on the motif 7.9%→35.8%, a 4.6× concentration after training).
  Spine untouched except one registry entry. `genomics` uv group (`logomaker`).
- **Weather head** (`src/heads/weather/`) — spatial field prediction, the second
  head against the unchanged contract (SPEC §3; a MINOR bump). Sweep axis
  operator {fno, unet} × resolution; metric field RMSE; viewer = predicted |
  truth | error triptych. FNO and U-Net are **hand-rolled in plain torch**
  (`torch.fft` spectral conv, differentiable on CPU) — no `neuraloperator`
  dependency (#9). Synthetic in-code deblur task (band-limited Gaussian random
  fields), smokes in seconds on CPU. Registered in the spine; **no spine code
  changed except one registry entry** — the conformance + tag-charset tests now
  auto-cover it. `weather` uv group (just `matplotlib`).
- **Stage 5 real-reclaim path (`ec2/` + board `--ec2-sweep`).** AWS FIS can't
  interrupt SageMaker *managed* spot (those instances live in a SageMaker service
  account), so a true FIS-driven reclaim runs training on a self-managed EC2 spot
  instance under an ASG (desired=1): `ec2/launch.sh` (launch template + ASG,
  user-data runs the same DLC + `train.py` resuming from S3), `ec2/fis.sh` (fires
  `aws:ec2:send-spot-instance-interruptions`), `ec2/teardown.sh`. The board gains
  an `--ec2-sweep` read path (`dashboard/ec2.go`) that renders EC2 instances from
  the **same §9 tags** — the contract is the seam across both executors. The
  instance interruption → `RESUMING` tile, then the ASG replacement resumes.
- **Cloud submit** (`scripts/submit.py`) — stage 3: one parameterized SageMaker
  training job writing the §9 job tags. Built against the installed sagemaker
  3.13.1 `ModelTrainer` API (the classic `PyTorch` estimator is gone — see #1).
  Dry-run by default; `--submit` required to spend.
- **Sweep fan-out** (`scripts/sweep.py`) — stage 4: cartesian product over
  `head.sweep_axes()`, one tagged `<Sweep>-NN` job per point. Dry-run by default.
- **Live board feed** (`dashboard/live.go`) — stage 4/5: read-only
  `ListTrainingJobs → ListTags → CloudWatch GetMetricData` (the #1 flow);
  `--sweep` enables live mode, otherwise the sample feed renders without AWS.
- **Spot/checkpoint resume** (molecular head) — stage 5: periodic checkpoints
  carry optimizer + epoch; a reclaimed job resumes from the synced checkpoint
  rather than restarting. Verified locally.
- **Result viewers** (stage 7): Streamlit collaborator viewer (`app/app.py`) and
  marimo operator cockpit (`analysis/explore.py`), both domain-blind via
  `head.viewer()`. New `viewers` uv dependency group.
- **Molecular head** (`src/heads/molecular/`) — ESOL log-solubility regression,
  the reference head built end to end (SPEC §3). Sweep axis featurization
  `{ecfp, graph, graph+3d}` × depth `{shallow, deep}`; metric RMSE; ECFP→MLP and
  graph→GCN models; `predict` + a viewer declaring the predicted-vs-actual
  scatter. Registered in the spine; passes the contract-conformance test.
  Smoke-runs locally in seconds (`--max-steps`); a full ECFP run reaches
  ~0.3 RMSE, a visible behavior delta. Vendored ESOL CSV for offline smoke.
  `data.py` shape tests + CLI arg-parsing tests added.
- Repo scaffold: `uv` project (`pyproject.toml`, `.python-version` 3.12.13),
  ruff + mypy, pytest.
- The head contract `src/heads/base.py` — the typed `Head` protocol every head
  implements (SPEC §2.1).
- Contract-conformance test (`tests/test_contract.py`) that every registered
  head must satisfy — the guard rail against special-casing the spine.
- Thin spine stubs: `cli.py` (`--domain` dispatch, `--max-steps`), `run.py`
  (the `Run` context), `metrics.py` (`MetricSink` fan-out), `registry.py`
  (lazy per-head loading), and the `train.py` entry point.
- `SPEC.md` §9 — the job-tag schema pinning the Python↔Go board contract.
- Per-head `uv` dependency groups; `molecular` group defined (RDKit, PyG core,
  pandas) ahead of the head landing.

[Unreleased]: https://github.com/scttfrdmn/aws-research-train-demo/commits/main

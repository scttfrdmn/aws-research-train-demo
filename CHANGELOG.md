# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

> Pre-1.0.0: the shape is still settling; a minor bump may break. The head
> contract in `src/heads/base.py` is the most load-bearing surface — a breaking
> change to it is called out loudly even pre-1.0.

## [Unreleased]

### Fixed
- **§9 tag values now satisfy SageMaker's charset.** `CreateTrainingJob` rejects
  tag values outside `[\p{L}\p{Z}\p{N}_.:/=+\-@]` — caught at real submit time.
  `tile_label` separator `·`→` / `; the `rmse↓` sort hint became a separate
  `MetricGoal=min` tag. Added a contract test guarding every head's tags.
- `submit.py` ships a per-head `requirements.txt` into the DLC and ignores
  `.venv/.git/data` from the source tarball; DLC default pinned to the
  installed-SDK-resolvable `pytorch 2.8 / py312` (the #1 report's 2.10/py313 is
  "Unsupported" by sagemaker 3.13.1).

### Added
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

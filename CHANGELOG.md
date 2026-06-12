# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

> Pre-1.0.0: the shape is still settling; a minor bump may break. The head
> contract in `src/heads/base.py` is the most load-bearing surface — a breaking
> change to it is called out loudly even pre-1.0.

## [Unreleased]

### Added
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

# Cloud setup — AWS resources for the demo

The AWS-side resources the cloud stages (3+) depend on. Created 2026-06-12 in
account `942542972736` (profile `aws`), region **`us-west-2`**.

> Planning/status lives in GitHub, not here — this file documents *provisioned
> infrastructure* so it's discoverable and reproducible. Read-only reference.

> **Status (2026-06-12): all resources below have been TORN DOWN** — the demo ran
> end to end (SageMaker sweep + EC2/FIS reclaim) and the account is a clean slate.
> The create/run/teardown commands here reproduce it from scratch.

## Resources

| Resource | Value |
|---|---|
| Region | `us-west-2` |
| S3 bucket | `aws-research-train-demo-942542972736-us-west-2` (public access blocked) |
| Data prefix | `s3://…/molecular/data/delaney-processed.csv` (ESOL, vendored copy) |
| Checkpoint prefix | `s3://…/<sweep-id>/<job-name>/checkpoints/` (per job) |
| Execution role | `arn:aws:iam::942542972736:role/aws-research-train-demo-exec` |
| Role inline policy | `training-access` — S3 (named bucket **and** the SDK's default `sagemaker-us-west-2-<acct>` session bucket), ECR pull, CloudWatch logs/metrics |
| DLC image | `pytorch-training:2.8-cpu-py312` (resolved via `image_uris.retrieve`; `-gpu-` variant when `--instance ml.g5.xlarge`) |

The role trusts `sagemaker.amazonaws.com` only and is scoped to this bucket —
least privilege, not `AmazonSageMakerFullAccess`.

### Instance choice: CPU, not GPU (account quota reality)

This account caps **`ml.g5.xlarge` at 1 concurrent training job** (both spot and
on-demand quota = 1), so the parallel sweep — the board's whole point — can't run
on g5 without a Service Quotas increase. The tiny ESOL models don't need a GPU
(they train on CPU in minutes — the local smoke already ran CPU/MPS), and CPU
instances have a quota of **30**.

Cheapest SageMaker *training* instances in us-west-2 (on-demand, verified via the
Pricing API 2026-06-12): **`ml.c7i.large` $0.107/hr**, `ml.m5.large`/`ml.m6i.large`
$0.115, `ml.c5.xlarge` $0.204. Graviton (`c7g`/`c8g`/`m8g`) and AMD (`c7a`/`m7a`)
are **not offered** for SageMaker training here, so they're not options despite
cheaper raw EC2. The scripts therefore **default to `ml.c7i.large`** (resolves a
`-cpu-` PyTorch 2.8 DLC; ~half the cost of the earlier c5.xlarge). Override with
`--instance ml.g5.xlarge` for a single GPU job. Verified: a 6-wide CPU spot sweep
ran fully parallel and completed.

### DLC version note (verify-first #1)

The installed `sagemaker` 3.13.1 `image_uris.retrieve` resolves **PyTorch 2.8 /
py312** as the newest in us-west-2; the report's `2.10/py313` snapshot returns
"Unsupported". `submit.py` / `sweep.py` default to `--framework-version 2.8
--py-version py312` accordingly. `retrieve` picks the `-cpu-` or `-gpu-` DLC
from the instance type, so the CPU default just works.

## How it was created

```bash
export AWS_PROFILE=aws AWS_REGION=us-west-2
BUCKET=aws-research-train-demo-942542972736-us-west-2

# bucket (+ block all public access) and data
aws s3api create-bucket --bucket "$BUCKET" --region us-west-2 \
  --create-bucket-configuration LocationConstraint=us-west-2
aws s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3 cp src/heads/molecular/data/delaney-processed.csv \
  "s3://$BUCKET/molecular/data/delaney-processed.csv"

# execution role: trust sagemaker + scoped inline policy
aws iam create-role --role-name aws-research-train-demo-exec \
  --assume-role-policy-document file://sm-trust.json
aws iam put-role-policy --role-name aws-research-train-demo-exec \
  --policy-name training-access --policy-document file://sm-policy.json
```

(`sm-trust.json` = trust `sagemaker.amazonaws.com`; `sm-policy.json` = the
`training-access` statements above.)

## Running the demo (stages 3–4)

Scripts are **dry-run by default**; `--submit` spends money. Always pass
`--region us-west-2` (the `aws` profile is us-west-2 but `AWS_REGION` env may be
something else).

```bash
export AWS_PROFILE=aws
BUCKET=aws-research-train-demo-942542972736-us-west-2
ROLE=arn:aws:iam::942542972736:role/aws-research-train-demo-exec

# Stage 3 — one job (dry-run first; add --submit to launch)
uv run --group cloud --group molecular python scripts/submit.py \
  --domain molecular --feat graph --depth deep \
  --sweep mol-esol-$(date +%Y%m%d)-a --seq 1 \
  --instance ml.g5.xlarge --s3-bucket "$BUCKET" --role-arn "$ROLE" \
  --region us-west-2 --submit

# Stage 4 — the 3×2 scientific grid (6 jobs), managed spot
uv run --group cloud --group molecular python scripts/sweep.py \
  --domain molecular --sweep mol-esol-$(date +%Y%m%d)-a --axes feat,depth \
  --instance ml.c7i.large --spot \
  --s3-bucket "$BUCKET" --role-arn "$ROLE" --region us-west-2 --submit

# Board — live, scoped to the sweep (read-only AWS)
cd dashboard && go run . --sweep mol-esol-$(date +%Y%m%d)-a --region us-west-2
```

## Stage 5 — real spot reclaim via FIS (self-managed EC2, Option A)

**Why not SageMaker managed spot:** verified 2026-06-12 — managed-spot training
instances run in a **SageMaker service-owned account**, so AWS FIS (which targets
EC2 instances in *your* account) cannot interrupt them, and there's no first-party
API to force a managed-spot interruption. To show a **real** reclaim +
checkpoint-resume, training runs on a plain EC2 spot instance we own. The §9 tags
are the seam: the instance carries the same `Sweep`/`Hypothesis`/`Metric`/… tags,
so the board renders it identically (see the board's `--ec2-sweep` mode).

| Resource | Value |
|---|---|
| EC2 instance role / profile | `aws-research-train-demo-ec2` (S3 rw to bucket, ECR pull, CloudWatch) |
| FIS experiment role | `aws-research-train-demo-fis` (`ec2:SendSpotInstanceInterruptions` + describe) |
| Launch template | `aws-research-train-demo-lt` (AL2023, spot, user-data runs the DLC) |
| Auto Scaling Group | `aws-research-train-demo-asg` (desired=1, **no** capacity-rebalance) |
| Instance type | `c7i.large` spot (plain EC2, no `ml.` prefix) |
| Checkpoint prefix | `s3://…/<sweep>/ec2/checkpoints/` |

The ASG holds desired=1: when the instance is interrupted, the ASG launches a
replacement that resumes from the S3 checkpoint. Capacity-rebalance is **off** so
AWS rebalance *recommendations* don't churn the instance before we fire FIS.

> **The ASG keeps a spot instance running until torn down** (unlike SageMaker
> jobs, which end on their own). ~$0.03/hr; run `ec2/teardown.sh` when done.

```bash
export AWS_PROFILE=aws AWS_REGION=us-west-2
ec2/launch.sh                      # IAM/profile already exist; creates LT + ASG + instance
# watch it (the board's second read path):
cd dashboard && go run . --ec2-sweep mol-esol-ec2spot --region us-west-2
# fire a REAL 2-minute interruption (billable, deliberate):
ec2/fis.sh                         # tile goes reclaim → RESUMING → green as ASG resumes
ec2/teardown.sh                    # remove ASG + LT + FIS templates (keeps shared bucket/role)
```

> **Curve/metric on the EC2 path:** the EC2 training run has no SageMaker
> log-scraper pushing metrics to CloudWatch, so the EC2 tile renders from
> instance **state** (the red→amber→green reclaim transition — which is the whole
> stage-5 point), not a live RMSE curve. The SageMaker path keeps the curve.

## Cost & control: SageMaker managed spot vs. self-managed EC2 spot

Both executors run the **same `train.py` / head / DLC / checkpoint logic**, and the
board renders both identically through the §9 tag contract (`--sweep` for
SageMaker, `--ec2-sweep` for EC2). The difference is purely **cost vs. control** —
there is a real `ml.` premium for the managed layer.

Prices for `c7i.large`, us-west-2 (Pricing/Spot APIs, verified 2026-06-12):

| | $/hr | vs EC2 on-demand |
|---|---|---|
| EC2 on-demand | $0.0893 | baseline |
| **SageMaker on-demand** (`ml.c7i.large` training) | $0.1070 | **+20%** |
| EC2 spot (cheapest AZ, observed) | ~$0.027 | −70% |
| SageMaker managed spot | ~$0.032 (≈ the +20% riding on the spot rate) | — |

**The ~20% premium buys managed convenience:** SageMaker detects the spot
interruption, requeues the job, restarts the container, and your `train.py`
resumes from `checkpoint_s3_uri` — **zero infra code**. You also get real
CloudWatch metric curves (the log-scraper), `BillableTimeInSeconds`, and job
identity preserved across reclaims. Just add `--spot` to `submit.py`/`sweep.py`.

**Self-managed EC2 spot (the `ec2/` path) trades that for control:** you own the
ASG + checkpoint-sync + interruption-watcher (everything in `ec2/`), but you get
~20% cheaper compute **and** the ability to trigger/test interruption behavior.

**What managed spot CANNOT do (why the `ec2/` path exists):** its instances run in
a SageMaker service account, so **AWS FIS cannot target them** and there is **no
2-minute interruption notice / countdown** exposed — you only see `Interrupted`
after the fact (the board maps that → `RESUMING`; see `statusFromSM` in `live.go`).
A real, on-demand FIS reclaim + the live `RECLAIM ⚠ 2:00` countdown require
owning the instance, hence self-managed EC2. To make a *managed-spot* reclaim
visible for a demo you'd wait for a natural reclaim or `StopTrainingJob` +
resubmit (deterministic, exercises resume, but not a true spot event).

**Guidance for the two audiences:**
- *Most researchers / move fast:* SageMaker managed spot — `train.py --spot`, ~20%
  more, you write none of the resilience.
- *Cost-sensitive / need interruption control:* self-managed EC2 spot — ~20%
  cheaper, FIS-testable, but you build (and own) the resilience.

## Teardown

Full teardown to a clean slate (this is exactly what was run 2026-06-12).

```bash
export AWS_PROFILE=aws AWS_REGION=us-west-2
BUCKET=aws-research-train-demo-942542972736-us-west-2

# 1. EC2/ASG/FIS path (stage 5) — ASG, launch template, FIS templates
ec2/teardown.sh

# 2. S3 bucket (data + checkpoints)
aws s3 rm "s3://$BUCKET" --recursive
aws s3api delete-bucket --bucket "$BUCKET" --region us-west-2

# 3. EC2 instance role + profile
aws iam remove-role-from-instance-profile --instance-profile-name aws-research-train-demo-ec2 --role-name aws-research-train-demo-ec2
aws iam delete-instance-profile --instance-profile-name aws-research-train-demo-ec2
aws iam delete-role-policy --role-name aws-research-train-demo-ec2 --policy-name training-access
aws iam delete-role --role-name aws-research-train-demo-ec2

# 4. FIS role
aws iam delete-role-policy --role-name aws-research-train-demo-fis --policy-name fis-spot-interrupt
aws iam delete-role --role-name aws-research-train-demo-fis

# 5. SageMaker execution role
aws iam delete-role-policy --role-name aws-research-train-demo-exec --policy-name training-access
aws iam delete-role --role-name aws-research-train-demo-exec
```

> Delete the ASG (step 1) **before** the EC2 instance role — the role is in use
> while the instance is alive. `ec2/teardown.sh` force-deletes the ASG, which
> terminates the instance first.

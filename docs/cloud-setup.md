# Cloud setup — AWS resources for the demo

The AWS-side resources the cloud stages (3+) depend on. Created 2026-06-12 in
account `942542972736` (profile `aws`), region **`us-west-2`**.

> Planning/status lives in GitHub, not here — this file documents *provisioned
> infrastructure* so it's discoverable and reproducible. Read-only reference.

## Resources

| Resource | Value |
|---|---|
| Region | `us-west-2` |
| S3 bucket | `aws-research-train-demo-942542972736-us-west-2` (public access blocked) |
| Data prefix | `s3://…/molecular/data/delaney-processed.csv` (ESOL, vendored copy) |
| Checkpoint prefix | `s3://…/<sweep-id>/<job-name>/checkpoints/` (per job) |
| Execution role | `arn:aws:iam::942542972736:role/aws-research-train-demo-exec` |
| Role inline policy | `training-access` — S3 (named bucket **and** the SDK's default `sagemaker-us-west-2-<acct>` session bucket), ECR pull, CloudWatch logs/metrics |
| DLC image | `pytorch-training:2.8-gpu-py312` (resolved via `image_uris.retrieve`) |

The role trusts `sagemaker.amazonaws.com` only and is scoped to this bucket —
least privilege, not `AmazonSageMakerFullAccess`.

### DLC version note (verify-first #1)

The installed `sagemaker` 3.13.1 `image_uris.retrieve` resolves **PyTorch 2.8 /
py312** as the newest in us-west-2; the report's `2.10/py313` snapshot returns
"Unsupported". `submit.py` / `sweep.py` default to `--framework-version 2.8
--py-version py312` accordingly. A GPU instance (default `ml.g5.xlarge`) is
required — the DLC is a `-gpu-` image.

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
  --instance ml.g5.xlarge --spot \
  --s3-bucket "$BUCKET" --role-arn "$ROLE" --region us-west-2 --submit

# Board — live, scoped to the sweep (read-only AWS)
cd dashboard && go run . --sweep mol-esol-$(date +%Y%m%d)-a --region us-west-2
```

## Teardown

```bash
export AWS_PROFILE=aws AWS_REGION=us-west-2
BUCKET=aws-research-train-demo-942542972736-us-west-2
aws s3 rm "s3://$BUCKET" --recursive
aws s3api delete-bucket --bucket "$BUCKET" --region us-west-2
aws iam delete-role-policy --role-name aws-research-train-demo-exec --policy-name training-access
aws iam delete-role --role-name aws-research-train-demo-exec
```

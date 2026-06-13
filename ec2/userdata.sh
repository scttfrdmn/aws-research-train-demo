#!/bin/bash
# Self-managed EC2 spot training bootstrap (stage 5, Option A — real FIS reclaim).
#
# Why this exists: SageMaker *managed* spot runs its instances in a SageMaker
# service account, which AWS FIS cannot target (verified 2026-06-12). To show a
# REAL spot interruption + checkpoint-resume, training runs here on a plain EC2
# spot instance we own — and the same §9 tags make it legible to the board.
#
# Lifecycle this script implements:
#   1. resume: s3 sync the checkpoint dir DOWN (empty on first launch)
#   2. train:  run the SAME PyTorch DLC + train.py we run on SageMaker, resuming
#   3. persist: a sidecar loop s3-syncs checkpoints UP every 30s
#   4. reclaim: a watcher polls IMDS for the 2-min interruption notice; on notice
#      it forces a final checkpoint sync so the ASG's replacement resumes cleanly
#
# Templated values ({{...}}) are filled by ec2/launch.sh before base64 encoding.
set -uxo pipefail

BUCKET="{{BUCKET}}"
SWEEP="{{SWEEP}}"
FEAT="{{FEAT}}"
DEPTH="{{DEPTH}}"
EPOCHS="{{EPOCHS}}"
REGION="{{REGION}}"
DLC="{{DLC}}"
CKPT_S3="s3://${BUCKET}/${SWEEP}/ec2/checkpoints/"
CKPT_LOCAL="/opt/ckpt"
DATA_S3="s3://${BUCKET}/molecular/data/"
DATA_LOCAL="/opt/data"

mkdir -p "$CKPT_LOCAL" "$DATA_LOCAL"
dnf install -y docker git >/dev/null 2>&1 || yum install -y docker git >/dev/null 2>&1
systemctl start docker

# source: clone the (public) repo so the container runs the SAME train.py + head
git clone --depth 1 https://github.com/scttfrdmn/aws-research-train-demo.git /opt/src || true

# 1. resume — pull any existing checkpoint + the dataset
aws s3 sync "$CKPT_S3" "$CKPT_LOCAL" --region "$REGION" || true
aws s3 sync "$DATA_S3" "$DATA_LOCAL/molecular/data/" --region "$REGION" || true

# 3. checkpoint persister (background): keep S3 current so a replacement resumes
(
  while true; do
    aws s3 sync "$CKPT_LOCAL" "$CKPT_S3" --region "$REGION" >/dev/null 2>&1 || true
    sleep 30
  done
) &
PERSIST_PID=$!

# 4. spot-interruption watcher (background): on the 2-min notice, force a sync
(
  TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
  while true; do
    CODE=$(curl -s -o /dev/null -w "%{http_code}" -H "X-aws-ec2-metadata-token: $TOKEN" \
      http://169.254.169.254/latest/meta-data/spot/instance-action)
    if [ "$CODE" = "200" ]; then
      echo "SPOT INTERRUPTION NOTICE — forcing final checkpoint sync"
      aws s3 sync "$CKPT_LOCAL" "$CKPT_S3" --region "$REGION" || true
      break
    fi
    sleep 5
  done
) &

# 2. login to ECR for the DLC, then train in-container resuming from $CKPT_LOCAL.
# train.py reads --checkpoint-dir via SM_CHECKPOINT_DIR and --data-dir via
# SM_CHANNEL_TRAINING — set both so the unchanged spine CLI resolves them.
ECR_ACCOUNT=$(echo "$DLC" | cut -d. -f1)
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ECR_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
docker pull "$DLC"

docker run --rm \
  -v "$CKPT_LOCAL":/opt/ml/checkpoints \
  -v "$DATA_LOCAL/molecular/data":/opt/ml/input/data/training \
  -e SM_CHECKPOINT_DIR=/opt/ml/checkpoints \
  -e SM_CHANNEL_TRAINING=/opt/ml/input/data/training \
  -v /opt/src:/opt/src \
  -w /opt/src \
  "$DLC" bash -c "
    pip install -q -r src/heads/molecular/requirements.txt &&
    python train.py --domain molecular --feat ${FEAT} --depth ${DEPTH} --epochs ${EPOCHS}
  "

# training finished without interruption — final sync, stop the persister
aws s3 sync "$CKPT_LOCAL" "$CKPT_S3" --region "$REGION" || true
kill $PERSIST_PID 2>/dev/null || true

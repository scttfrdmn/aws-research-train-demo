#!/bin/bash
# Fire a REAL spot interruption at the self-managed training instance via AWS FIS
# (stage 5, Option A). aws:ec2:send-spot-instance-interruptions delivers the same
# 2-minute interruption notice EC2 sends on a genuine reclaim, then stops the
# instance — so the ASG launches a replacement that resumes from the S3
# checkpoint. The board tile goes red (reclaim) → amber (RESUMING) → green.
#
# This is the BILLABLE, irreversible action. Run it deliberately.
# Usage: ec2/fis.sh
set -euo pipefail

REGION="${REGION:-us-west-2}"
SWEEP="${SWEEP:-mol-esol-ec2spot}"
FIS_ROLE="arn:aws:iam::942542972736:role/aws-research-train-demo-fis"

echo "Resolving the running spot instance for sweep ${SWEEP}…"
IID=$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:Sweep,Values=${SWEEP}" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)
if [ "$IID" = "None" ] || [ -z "$IID" ]; then
  echo "No running instance tagged Sweep=${SWEEP}. Launch first with ec2/launch.sh." >&2
  exit 1
fi
echo "Target instance: $IID"

# FIS experiment template: interrupt the resolved instance in 2 minutes.
TEMPLATE=$(cat <<JSON
{
  "description": "spot interruption for aws-research-train-demo (${SWEEP})",
  "roleArn": "${FIS_ROLE}",
  "stopConditions": [{"source": "none"}],
  "targets": {
    "t1": {
      "resourceType": "aws:ec2:spot-instance",
      "resourceArns": ["arn:aws:ec2:${REGION}:942542972736:instance/${IID}"],
      "selectionMode": "ALL"
    }
  },
  "actions": {
    "interrupt": {
      "actionId": "aws:ec2:send-spot-instance-interruptions",
      "parameters": {"durationBeforeInterruption": "PT2M"},
      "targets": {"SpotInstances": "t1"}
    }
  },
  "tags": {"project": "aws-research-train-demo"}
}
JSON
)

echo "Creating FIS experiment template…"
TID=$(aws fis create-experiment-template --region "$REGION" \
  --cli-input-json "$TEMPLATE" --query 'experimentTemplate.id' --output text)
echo "template: $TID"

echo "Starting experiment (real 2-minute interruption notice → reclaim)…"
EID=$(aws fis start-experiment --region "$REGION" \
  --experiment-template-id "$TID" --query 'experiment.id' --output text)
echo "experiment: $EID"
echo
echo "Watch the board: the tile for ${IID} should go reclaim → RESUMING → green"
echo "as the ASG replaces the instance and it resumes from the S3 checkpoint."
echo "Experiment template ${TID} is left for teardown (ec2/teardown.sh removes it)."

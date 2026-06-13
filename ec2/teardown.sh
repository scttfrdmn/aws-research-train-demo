#!/bin/bash
# Tear down the Option-A self-managed spot infra. Leaves the shared bucket/role
# from docs/cloud-setup.md alone — only removes the EC2/ASG/FIS pieces.
# Usage: ec2/teardown.sh
set -uo pipefail

REGION="${REGION:-us-west-2}"
ASG_NAME="aws-research-train-demo-asg"
LT_NAME="aws-research-train-demo-lt"

echo "Deleting ASG ${ASG_NAME} (force, terminates the instance)…"
aws autoscaling delete-auto-scaling-group --region "$REGION" \
  --auto-scaling-group-name "$ASG_NAME" --force-delete 2>&1 || true

echo "Deleting launch template ${LT_NAME}…"
aws ec2 delete-launch-template --region "$REGION" \
  --launch-template-name "$LT_NAME" 2>&1 || true

echo "Deleting FIS experiment templates for this project…"
for TID in $(aws fis list-experiment-templates --region "$REGION" \
  --query "experimentTemplates[?tags.project=='aws-research-train-demo'].id" --output text 2>/dev/null); do
  aws fis delete-experiment-template --region "$REGION" --id "$TID" 2>&1 || true
  echo "  deleted $TID"
done

echo "Done. To also remove IAM roles (only if you're fully done):"
echo "  aws iam remove-role-from-instance-profile --instance-profile-name aws-research-train-demo-ec2 --role-name aws-research-train-demo-ec2"
echo "  aws iam delete-instance-profile --instance-profile-name aws-research-train-demo-ec2"
echo "  aws iam delete-role-policy --role-name aws-research-train-demo-ec2 --policy-name training-access && aws iam delete-role --role-name aws-research-train-demo-ec2"
echo "  aws iam delete-role-policy --role-name aws-research-train-demo-fis --policy-name fis-spot-interrupt && aws iam delete-role --role-name aws-research-train-demo-fis"

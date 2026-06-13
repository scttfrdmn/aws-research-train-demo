#!/bin/bash
# Launch the self-managed EC2 spot training instance via an Auto Scaling Group
# (stage 5, Option A). The ASG (desired=1, capacity-rebalance) is what makes the
# reclaim survivable: when FIS interrupts the spot instance, the ASG launches a
# replacement that resumes from the S3 checkpoint.
#
# The instance is tagged with the §9 keys so the board's EC2 read path renders it
# exactly like a SageMaker tile. Idempotent-ish: re-run after ./teardown.sh.
#
# Usage: ec2/launch.sh   (env-overridable vars below)
set -euo pipefail

REGION="${REGION:-us-west-2}"
BUCKET="${BUCKET:-aws-research-train-demo-942542972736-us-west-2}"
SWEEP="${SWEEP:-mol-esol-ec2spot}"
FEAT="${FEAT:-graph}"
DEPTH="${DEPTH:-deep}"
EPOCHS="${EPOCHS:-5000}"          # long, so there's a window to interrupt
INSTANCE_TYPE="${INSTANCE_TYPE:-c7i.large}"   # plain EC2 type (no ml. prefix)
AMI="${AMI:-ami-0d45a4eba03d1e2cf}"           # AL2023 x86_64, us-west-2
SUBNET="${SUBNET:-subnet-0376742606a975d27}"  # default-VPC public subnet (us-west-2b)
SG="${SG:-sg-5059b179}"                       # default VPC security group
PROFILE_ARN="arn:aws:iam::942542972736:instance-profile/aws-research-train-demo-ec2"
DLC="${DLC:-763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.8-cpu-py312}"
LT_NAME="aws-research-train-demo-lt"
ASG_NAME="aws-research-train-demo-asg"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "Templating user-data…"
UD=$(sed \
  -e "s|{{BUCKET}}|${BUCKET}|g" -e "s|{{SWEEP}}|${SWEEP}|g" \
  -e "s|{{FEAT}}|${FEAT}|g"     -e "s|{{DEPTH}}|${DEPTH}|g" \
  -e "s|{{EPOCHS}}|${EPOCHS}|g" -e "s|{{REGION}}|${REGION}|g" \
  -e "s|{{DLC}}|${DLC}|g" "${HERE}/userdata.sh" | base64)

echo "Creating launch template ${LT_NAME}…"
# §9 tags on the instance — the board's EC2 path reads these (Domain display-only).
TAGSPEC='{"ResourceType":"instance","Tags":[
  {"Key":"Sweep","Value":"'"${SWEEP}"'"},
  {"Key":"Hypothesis","Value":"feat='"${FEAT}"' / depth='"${DEPTH}"'"},
  {"Key":"Metric","Value":"rmse"},
  {"Key":"MetricGoal","Value":"min"},
  {"Key":"Domain","Value":"molecular"},
  {"Key":"Instance","Value":"'"${INSTANCE_TYPE}"'"},
  {"Key":"Spot","Value":"true"},
  {"Key":"project","Value":"aws-research-train-demo"}]}'

aws ec2 create-launch-template --region "$REGION" \
  --launch-template-name "$LT_NAME" \
  --launch-template-data '{
    "ImageId":"'"$AMI"'",
    "InstanceType":"'"$INSTANCE_TYPE"'",
    "IamInstanceProfile":{"Arn":"'"$PROFILE_ARN"'"},
    "InstanceMarketOptions":{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}},
    "NetworkInterfaces":[{"DeviceIndex":0,"AssociatePublicIpAddress":true,"Groups":["'"$SG"'"],"SubnetId":"'"$SUBNET"'"}],
    "UserData":"'"$UD"'",
    "TagSpecifications":['"$TAGSPEC"']
  }' --query 'LaunchTemplate.LaunchTemplateId' --output text

# desired=1 is what makes a reclaim survivable: terminate the instance (FIS) and
# the ASG launches a replacement that resumes from the S3 checkpoint. We do NOT
# enable capacity-rebalance — that proactively cycles instances on AWS rebalance
# *recommendations*, which churns the demo before we deliberately fire FIS.
echo "Creating Auto Scaling Group ${ASG_NAME} (desired=1, no capacity-rebalance)…"
aws autoscaling create-auto-scaling-group --region "$REGION" \
  --auto-scaling-group-name "$ASG_NAME" \
  --launch-template "LaunchTemplateName=${LT_NAME},Version=\$Latest" \
  --min-size 1 --max-size 1 --desired-capacity 1 \
  --no-capacity-rebalance \
  --vpc-zone-identifier "$SUBNET" \
  --tags "Key=project,Value=aws-research-train-demo,PropagateAtLaunch=false"

echo "Done. ASG ${ASG_NAME} is bringing up one ${INSTANCE_TYPE} spot instance."
echo "Watch it on the board:  cd dashboard && go run . --ec2-sweep ${SWEEP} --region ${REGION}"
echo "Fire a real reclaim:    ec2/fis.sh"

// EC2 read path for the board — the self-managed-spot side of stage 5 (Option A).
//
// SageMaker managed-spot instances live in a SageMaker service account that AWS
// FIS can't target, so the "survive a real reclaim" demo runs training on a
// plain EC2 spot instance we own (see ec2/). That instance is NOT a SageMaker
// job, so the board reads it a different way — but through the SAME §9 tags, so
// a tile looks identical. This file is that second read path.
//
// All calls are read-only: DescribeInstances + CloudWatch GetMetricData.
package main

import (
	"context"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/cloudwatch"
	"github.com/aws/aws-sdk-go-v2/service/ec2"
	ec2types "github.com/aws/aws-sdk-go-v2/service/ec2/types"
)

type ec2Client struct {
	ec2 *ec2.Client
	cw  *cloudwatch.Client
}

func newEC2Client(ctx context.Context, region string) (*ec2Client, error) {
	opts := []func(*config.LoadOptions) error{}
	if region != "" {
		opts = append(opts, config.WithRegion(region))
	}
	cfg, err := config.LoadDefaultConfig(ctx, opts...)
	if err != nil {
		return nil, err
	}
	return &ec2Client{ec2: ec2.NewFromConfig(cfg), cw: cloudwatch.NewFromConfig(cfg)}, nil
}

// ec2StateToBoard maps EC2 instance + spot state to the board's tile states.
// A pending spot interruption (set by a real reclaim or FIS) → RESUMING, which
// is the whole point: the board shows the reclaim, then the ASG's replacement
// comes up PENDING → IN_PROGRESS → COMPLETED as it resumes from S3.
func ec2StateToBoard(state ec2types.InstanceStateName, interruptionPending bool) string {
	if interruptionPending {
		return "RESUMING"
	}
	switch state {
	case ec2types.InstanceStateNameRunning:
		return "IN_PROGRESS"
	case ec2types.InstanceStateNamePending:
		return "PENDING"
	case ec2types.InstanceStateNameShuttingDown, ec2types.InstanceStateNameStopping,
		ec2types.InstanceStateNameStopped, ec2types.InstanceStateNameTerminated:
		// instance going away — a reclaim in progress reads as RESUMING above;
		// otherwise treat as completed/cycling.
		return "RESUMING"
	default:
		return "PENDING"
	}
}

// fetchEC2Sweep builds the board feed from EC2 instances tagged Sweep=<sweep>.
func (ec *ec2Client) fetchEC2Sweep(ctx context.Context, sweep string) (feed, error) {
	out, err := ec.ec2.DescribeInstances(ctx, &ec2.DescribeInstancesInput{
		Filters: []ec2types.Filter{
			{Name: aws.String("tag:Sweep"), Values: []string{sweep}},
			{Name: aws.String("instance-state-name"),
				Values: []string{"pending", "running", "shutting-down", "stopping", "stopped"}},
		},
	})
	if err != nil {
		return feed{}, err
	}

	jobs := []Job{}
	for _, r := range out.Reservations {
		for _, inst := range r.Instances {
			id := aws.ToString(inst.InstanceId)
			job := Job{ID: id, Spot: inst.InstanceLifecycle == ec2types.InstanceLifecycleTypeSpot}

			interruptionPending := inst.SpotInstanceRequestId != nil &&
				instanceHasInterruptionTag(inst)
			var stateName ec2types.InstanceStateName
			if inst.State != nil {
				stateName = inst.State.Name
			}
			job.Status = ec2StateToBoard(stateName, interruptionPending)

			applyEC2Tags(&job, inst.Tags)
			if job.MetricName != "" && job.Status != "PENDING" {
				job.Curve = ec.metricSeries(ctx, id, job.MetricName)
				if len(job.Curve) > 0 {
					v := job.Curve[len(job.Curve)-1]
					job.MetricValue = &v
				}
			}
			jobs = append(jobs, job)
		}
	}
	return feed{Sweep: sweep, Jobs: jobs}, nil
}

// instanceHasInterruptionTag reports whether a reclaim is in flight. EC2 doesn't
// expose the 2-minute notice via DescribeInstances, so we infer it from the
// instance leaving the running state (shutting-down/stopping) — the userdata
// watcher has by then forced a final checkpoint sync.
func instanceHasInterruptionTag(inst ec2types.Instance) bool {
	if inst.State == nil {
		return false
	}
	switch inst.State.Name {
	case ec2types.InstanceStateNameShuttingDown, ec2types.InstanceStateNameStopping:
		return true
	default:
		return false
	}
}

// applyEC2Tags maps §9 tags onto the tile — identical keys to the SageMaker
// path (live.go). The board stays domain-blind: Domain is read, never branched.
func applyEC2Tags(job *Job, tags []ec2types.Tag) {
	for _, t := range tags {
		k, v := aws.ToString(t.Key), aws.ToString(t.Value)
		switch k {
		case "Hypothesis":
			job.Hypothesis = v
		case "Metric":
			job.MetricName = v
		case "Domain":
			job.Domain = v
		case "Instance":
			job.Instance = v
		case "Spot":
			job.Spot = v == "true"
		}
	}
}

// metricSeries pulls the metric series the training loop publishes to CloudWatch
// (the userdata runs the same train.py; the metric carries the instance id as
// the TrainingJobName-equivalent dimension). Best-effort; nil on any error.
func (ec *ec2Client) metricSeries(ctx context.Context, instanceID, metric string) []float64 {
	// The EC2 path's training publishes under the same namespace with the
	// instance id as dimension; if absent, the tile still renders from state.
	return nil // populated once the EC2 metric publish is wired (see notes)
}

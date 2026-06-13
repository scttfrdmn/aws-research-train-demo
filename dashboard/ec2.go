// EC2 read path for the board — the self-managed-spot side of stage 5 (Option A).
//
// SageMaker managed-spot instances live in a SageMaker service account that AWS
// FIS can't target, so the "survive a real reclaim" demo runs training on a
// plain EC2 spot instance we own (see ec2/). That instance is NOT a SageMaker
// job, so the board reads it a different way — but through the SAME §9 tags, so
// a tile looks identical. This file is that second read path.
//
// All calls are read-only: DescribeInstances + S3 GetObject (the checkpoint
// meta.json, which doubles as the progress record since a self-managed instance
// has no SageMaker log-scraper feeding CloudWatch).
package main

import (
	"context"
	"encoding/json"
	"io"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/ec2"
	ec2types "github.com/aws/aws-sdk-go-v2/service/ec2/types"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

type ec2Client struct {
	ec2    *ec2.Client
	s3     *s3.Client
	bucket string // checkpoint bucket; meta.json holds live progress
	mu     sync.Mutex
	state  map[string]*sweepState // per-sweep accumulated view (survives swaps)
}

// sweepState is what makes the EC2 tile stable across instance swaps: the ASG
// holds desired=1, so a sweep is ONE logical training job even as the backing
// instance is reclaimed and replaced. The board accumulates the curve here
// (self-managed EC2 has no CloudWatch series) and remembers a reclaim so the
// tile shows RESUMING through the gap instead of vanishing.
type sweepState struct {
	curve         []float64
	lastEpoch     int
	reclaimMarker int        // curve index where the last reclaim happened (-1 = none)
	sawInstance   bool       // have we ever seen a running instance for this sweep?
	runningSec    float64    // accumulated RUNNING seconds (billed; excludes reclaim gaps)
	lastTick      *time.Time // wall-clock of the previous poll, to add the delta
	t0            *time.Time // first time we saw an instance — wall-clock t0 for effective rate
	noticeAt      *time.Time // when we first saw the 2-min interruption notice (countdown t0)
}

// spotGraceSec is EC2's spot interruption notice window: the instance keeps
// running this long after the notice before it's reclaimed.
const spotGraceSec = 120

// approxSpotUSDPerHour is the rough spot price for the cost meter. c7i.large
// on-demand is $0.107/hr (verified via Pricing API, us-west-2, 2026-06-12);
// spot typically runs ~60-70% off, so ~$0.04/hr. The meter is a live "real
// money is being spent" indicator, not an invoice — approximate is fine.
const approxSpotUSDPerHour = 0.04

func newEC2Client(ctx context.Context, region, bucket string) (*ec2Client, error) {
	opts := []func(*config.LoadOptions) error{}
	if region != "" {
		opts = append(opts, config.WithRegion(region))
	}
	cfg, err := config.LoadDefaultConfig(ctx, opts...)
	if err != nil {
		return nil, err
	}
	return &ec2Client{
		ec2:    ec2.NewFromConfig(cfg),
		s3:     s3.NewFromConfig(cfg),
		bucket: bucket,
		state:  map[string]*sweepState{},
	}, nil
}

// checkpointMeta mirrors the head's meta.json (src/heads/molecular/head.py).
type checkpointMeta struct {
	Epoch  int     `json:"epoch"`
	Total  int     `json:"total"`
	Metric string  `json:"metric"`
	RMSE   float64 `json:"rmse"`
}

// progress reads <bucket>/<sweep>/ec2/checkpoints/meta.json — the live progress
// record the training loop writes every few epochs. Returns nil if absent (the
// instance is still booting / hasn't checkpointed yet), so the tile still
// renders from instance state.
func (ec *ec2Client) progress(ctx context.Context, sweep string) *checkpointMeta {
	if ec.bucket == "" {
		return nil
	}
	key := sweep + "/ec2/checkpoints/meta.json"
	out, err := ec.s3.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(ec.bucket), Key: aws.String(key),
	})
	if err != nil {
		return nil
	}
	defer out.Body.Close()
	body, err := io.ReadAll(out.Body)
	if err != nil {
		return nil
	}
	var m checkpointMeta
	if json.Unmarshal(body, &m) != nil {
		return nil
	}
	return &m
}

// noticeFired reports whether a running instance's spot request has received an
// interruption notice — i.e. we're inside the 2-minute grace window (FIS or a
// real reclaim). The spot request flips to a terminate/stop status code while
// the instance is still running; that gap is the window.
func (ec *ec2Client) noticeFired(ctx context.Context, inst *ec2types.Instance) bool {
	if inst == nil || inst.SpotInstanceRequestId == nil {
		return false
	}
	out, err := ec.ec2.DescribeSpotInstanceRequests(ctx, &ec2.DescribeSpotInstanceRequestsInput{
		SpotInstanceRequestIds: []string{aws.ToString(inst.SpotInstanceRequestId)},
	})
	if err != nil || len(out.SpotInstanceRequests) == 0 {
		return false
	}
	code := ""
	if s := out.SpotInstanceRequests[0].Status; s != nil {
		code = aws.ToString(s.Code)
	}
	// these codes mean "reclaim committed" — during the grace window the
	// instance is still running, which is exactly what we want to surface.
	switch code {
	case "instance-terminated-by-experiment", "marked-for-termination",
		"instance-terminated-by-price", "instance-terminated-no-capacity",
		"instance-terminated-capacity-oversubscribed", "instance-stopped-by-experiment":
		return true
	default:
		return false
	}
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

	// Pick the most-relevant instance for this sweep: a running one if any, else
	// the latest by launch time (so a reclaim-in-progress instance still shows).
	var current *ec2types.Instance
	for i := range out.Reservations {
		for j := range out.Reservations[i].Instances {
			inst := &out.Reservations[i].Instances[j]
			if current == nil {
				current = inst
				continue
			}
			// prefer running; otherwise the more recently launched
			curRunning := current.State != nil && current.State.Name == ec2types.InstanceStateNameRunning
			instRunning := inst.State != nil && inst.State.Name == ec2types.InstanceStateNameRunning
			if instRunning && !curRunning {
				current = inst
			} else if instRunning == curRunning && inst.LaunchTime != nil &&
				(current.LaunchTime == nil || inst.LaunchTime.After(*current.LaunchTime)) {
				current = inst
			}
		}
	}

	meta := ec.progress(ctx, sweep)

	ec.mu.Lock()
	st := ec.state[sweep]
	if st == nil {
		st = &sweepState{reclaimMarker: -1}
		ec.state[sweep] = st
	}

	// One stable tile per sweep — the ASG holds desired=1, so a sweep is ONE
	// logical training job across instance swaps. The tile never disappears; it
	// shows the reclaim instead (mirrors the mockup's run-05).
	job := Job{ID: sweep, Spot: true}
	if meta != nil {
		applyMeta(&job, meta)
		// accumulate the curve as epochs advance (no CloudWatch on self-managed)
		if meta.RMSE != 0 && meta.Epoch != st.lastEpoch {
			st.curve = append(st.curve, meta.RMSE)
			st.lastEpoch = meta.Epoch
		}
	}

	now := time.Now()
	currentRunning := current != nil && current.State != nil &&
		current.State.Name == ec2types.InstanceStateNameRunning
	// Are we inside the 2-minute spot grace window? Only worth the extra API
	// call while an instance is actually running.
	inWindow := currentRunning && ec.noticeFired(ctx, current)
	if !inWindow {
		st.noticeAt = nil // window cleared (resumed, or never started)
	} else if st.noticeAt == nil {
		st.noticeAt = &now // first poll that saw the notice → countdown t0
	}

	reclaiming := current != nil && instanceHasInterruptionTag(*current)
	switch {
	case inWindow:
		// THE 2-MINUTE WINDOW: notice fired, instance still running + training.
		// Distinct red RECLAIM state with a live countdown (mockup's "reclaim").
		st.sawInstance = true
		applyEC2Tags(&job, current.Tags)
		job.Status = "RECLAIM"
		job.Instance = instanceTypeTag(current.Tags, job.Instance)
		left := spotGraceSec - int(now.Sub(*st.noticeAt).Seconds())
		if left < 0 {
			left = 0
		}
		job.ReclaimSecLeft = &left
	case current == nil && st.sawInstance:
		// instance gone, replacement not yet visible — the reclaim gap. Hold the
		// tile in RESUMING rather than dropping it.
		job.Status = "RESUMING"
	case current == nil:
		job.Status = "PENDING"
	default:
		st.sawInstance = true
		applyEC2Tags(&job, current.Tags)
		var stateName ec2types.InstanceStateName
		if current.State != nil {
			stateName = current.State.Name
		}
		job.Status = ec2StateToBoard(stateName, reclaiming)
		job.Instance = instanceTypeTag(current.Tags, job.Instance)
	}

	if (job.Status == "RECLAIM" || job.Status == "RESUMING") && st.reclaimMarker < 0 {
		st.reclaimMarker = len(st.curve) // mark where the reclaim cut the curve
	} else if job.Status == "IN_PROGRESS" {
		st.reclaimMarker = -1 // resumed cleanly; clear the marker
	}

	job.Curve = append([]float64(nil), st.curve...)
	if st.reclaimMarker >= 0 && st.reclaimMarker <= len(job.Curve) {
		rx := float64(st.reclaimMarker) / float64(max(len(job.Curve)-1, 1)) * 200
		job.ReclaimAt = &rx
	}

	// Meter: accumulate time while an instance is actually running — including
	// the RECLAIM window (still billed) but NOT the RESUMING gap (no instance)
	// or PENDING boot. t0 is the first sighting (wall-clock); the effective rate
	// = billed cost / wall-clock, which DROPS on every reclaim gap (you spanned
	// more time than you paid for — the spot-savings story).
	//
	// Cold-start seeding: the board is a stateless viewer, so on (re)start we
	// don't reset to 0 — we seed from the running instance's authoritative EC2
	// LaunchTime, so a reload immediately shows the instance's real age. (Billed
	// time from already-terminated prior instances can't be recovered after a
	// restart; we count from the live instance, which is the honest floor.)
	if st.lastTick == nil && current != nil && current.LaunchTime != nil &&
		(job.Status == "IN_PROGRESS" || job.Status == "RECLAIM") {
		st.runningSec = now.Sub(*current.LaunchTime).Seconds()
		st.t0 = current.LaunchTime
	}
	if st.t0 == nil && st.sawInstance {
		st.t0 = &now
	}
	if (job.Status == "IN_PROGRESS" || job.Status == "RECLAIM") && st.lastTick != nil {
		st.runningSec += now.Sub(*st.lastTick).Seconds()
	}
	st.lastTick = &now
	job.ElapsedSec = int(st.runningSec)
	job.CostUSD = st.runningSec / 3600.0 * approxSpotUSDPerHour
	if st.t0 != nil {
		wall := now.Sub(*st.t0).Seconds()
		job.WallSec = int(wall)
		if wall > 0 {
			job.EffUSDPerHr = job.CostUSD / (wall / 3600.0)
		}
	}
	ec.mu.Unlock()

	return feed{Sweep: sweep, Jobs: []Job{job}}, nil
}

// applyMeta sets step/total/metric from the checkpoint progress record.
func applyMeta(job *Job, m *checkpointMeta) {
	job.Step = m.Epoch
	job.TotalSteps = m.Total
	job.MetricName = m.Metric
	if m.RMSE != 0 {
		v := m.RMSE
		job.MetricValue = &v
	}
}

// instanceTypeTag prefers the Instance tag value, falling back to a default.
func instanceTypeTag(tags []ec2types.Tag, fallback string) string {
	for _, t := range tags {
		if aws.ToString(t.Key) == "Instance" {
			return aws.ToString(t.Value)
		}
	}
	return fallback
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

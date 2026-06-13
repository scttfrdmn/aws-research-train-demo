// Live AWS feed for the board — the read-only control-plane path (SPEC §5.1, §9).
//
// Per the verify-first finding (#1): tags are NOT returned by ListTrainingJobs
// or DescribeTrainingJob, so the flow is:
//
//	ListTrainingJobs(NameContains=<sweep>)   -> scope the sweep by name prefix (§9.3)
//	  per job: ListTags(ResourceArn)         -> the §9 tags (Hypothesis/Metric/...)
//	  per job: DescribeTrainingJob           -> status + FinalMetricDataList
//	  (optional) CloudWatch GetMetricData    -> the live series for the sparkline
//
// Every call is read-only. The board never writes, never touches compute.
package main

import (
	"context"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/cloudwatch"
	cwtypes "github.com/aws/aws-sdk-go-v2/service/cloudwatch/types"
	"github.com/aws/aws-sdk-go-v2/service/sagemaker"
	smtypes "github.com/aws/aws-sdk-go-v2/service/sagemaker/types"
)

// cwNamespace is where SageMaker publishes training-job metrics (#1 §4).
const cwNamespace = "/aws/sagemaker/TrainingJobs"

// liveClient holds the read-only AWS clients. Constructed once at startup.
type liveClient struct {
	sm *sagemaker.Client
	cw *cloudwatch.Client
}

func newLiveClient(ctx context.Context, region string) (*liveClient, error) {
	opts := []func(*config.LoadOptions) error{}
	if region != "" {
		opts = append(opts, config.WithRegion(region))
	}
	cfg, err := config.LoadDefaultConfig(ctx, opts...)
	if err != nil {
		return nil, err
	}
	return &liveClient{sm: sagemaker.NewFromConfig(cfg), cw: cloudwatch.NewFromConfig(cfg)}, nil
}

// statusFromSM maps SageMaker (secondary) status to the board's tile states.
func statusFromSM(primary smtypes.TrainingJobStatus, secondary smtypes.SecondaryStatus) string {
	switch primary {
	case smtypes.TrainingJobStatusCompleted:
		return "COMPLETED"
	case smtypes.TrainingJobStatusInProgress:
		// Managed-spot interruption surfaces as a secondary status; the job is
		// rejoining from its checkpoint — render it as RESUMING (mockup amber).
		if secondary == smtypes.SecondaryStatusInterrupted ||
			secondary == smtypes.SecondaryStatusRestarting {
			return "RESUMING"
		}
		if secondary == smtypes.SecondaryStatusStarting {
			return "PENDING"
		}
		return "IN_PROGRESS"
	case smtypes.TrainingJobStatusStopping, smtypes.TrainingJobStatusStopped:
		return "COMPLETED"
	default:
		return "PENDING"
	}
}

// fetchSweep builds the board feed for one sweep (job-name prefix = sweep id).
func (lc *liveClient) fetchSweep(ctx context.Context, sweep string) (feed, error) {
	out, err := lc.sm.ListTrainingJobs(ctx, &sagemaker.ListTrainingJobsInput{
		NameContains: aws.String(sweep),
		MaxResults:   aws.Int32(100),
		SortBy:       smtypes.SortByCreationTime,
		SortOrder:    smtypes.SortOrderAscending,
	})
	if err != nil {
		return feed{}, err
	}

	jobs := make([]Job, 0, len(out.TrainingJobSummaries))
	for _, s := range out.TrainingJobSummaries {
		name := aws.ToString(s.TrainingJobName)
		job := Job{ID: strings.TrimPrefix(name, sweep+"-")}

		// Describe → status + final metric (no tags here, per #1).
		desc, derr := lc.sm.DescribeTrainingJob(ctx, &sagemaker.DescribeTrainingJobInput{
			TrainingJobName: aws.String(name),
		})
		if derr == nil {
			job.Status = statusFromSM(desc.TrainingJobStatus, desc.SecondaryStatus)
			job.Spot = aws.ToBool(desc.EnableManagedSpotTraining)
			if len(desc.FinalMetricDataList) > 0 {
				v := float64(aws.ToFloat32(desc.FinalMetricDataList[0].Value))
				job.MetricValue = &v
			}
		}

		// Tags carry the §9 head-derived values (separate ListTags call).
		if arn := aws.ToString(s.TrainingJobArn); arn != "" {
			if tg, terr := lc.sm.ListTags(ctx, &sagemaker.ListTagsInput{
				ResourceArn: aws.String(arn),
			}); terr == nil {
				applyTags(&job, tg.Tags)
			}
		}

		// Live series for the sparkline (best-effort; tile renders without it).
		if job.MetricName != "" && job.Status != "PENDING" {
			job.Curve = lc.metricSeries(ctx, name, job.MetricName)
		}
		jobs = append(jobs, job)
	}
	return feed{Sweep: sweep, Jobs: jobs}, nil
}

// applyTags maps the §9 tags onto the tile fields. The board reads Domain but
// never branches on it — it stays domain-blind.
func applyTags(job *Job, tags []smtypes.Tag) {
	for _, t := range tags {
		k, v := aws.ToString(t.Key), aws.ToString(t.Value)
		switch k {
		case "Hypothesis":
			job.Hypothesis = v
		case "Metric":
			job.MetricName = v // sort direction is the separate MetricGoal tag
		case "Domain":
			job.Domain = v
		case "Instance":
			job.Instance = v
		case "Spot":
			job.Spot = v == "true"
		}
	}
}

// metricSeries pulls the named metric's recent datapoints from CloudWatch,
// keyed by the Host dimension (#1 §4). Returns nil on any error — the tile
// renders fine without a curve.
func (lc *liveClient) metricSeries(ctx context.Context, jobName, metric string) []float64 {
	end := time.Now()
	start := end.Add(-6 * time.Hour)
	out, err := lc.cw.GetMetricData(ctx, &cloudwatch.GetMetricDataInput{
		StartTime: aws.Time(start),
		EndTime:   aws.Time(end),
		MetricDataQueries: []cwtypes.MetricDataQuery{{
			Id: aws.String("m"),
			MetricStat: &cwtypes.MetricStat{
				Metric: &cwtypes.Metric{
					Namespace:  aws.String(cwNamespace),
					MetricName: aws.String(metric),
					Dimensions: []cwtypes.Dimension{{
						Name:  aws.String("Host"),
						Value: aws.String(jobName + "/algo-1"),
					}},
				},
				Period: aws.Int32(60),
				Stat:   aws.String("Average"),
			},
			ReturnData: aws.Bool(true),
		}},
		ScanBy: cwtypes.ScanByTimestampAscending,
	})
	if err != nil || len(out.MetricDataResults) == 0 {
		return nil
	}
	vals := out.MetricDataResults[0].Values
	if len(vals) == 0 {
		return nil
	}
	return vals
}

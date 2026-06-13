// Command dashboard is the Training Mission Control board: a read-only
// control-plane client (SPEC §5.1). It serves the approved board view at "/" and
// a JSON job feed at "/api/jobs" that the page polls every few seconds.
//
// The board never imports head code. Every value it renders comes from
// SageMaker job *tags* (SPEC §9) — Hypothesis, Metric, Domain, Instance, Spot —
// which scripts/sweep.py writes at submit time. That is what lets the board be
// Go while the heads stay Python.
//
// This scaffold ships the HTTP surface and the §9-shaped feed. The live AWS
// reads (ListTrainingJobs → per-job ListTags → CloudWatch GetMetricData, all
// read-only; see issue #1) land in build-order (c); until then the feed serves
// sample data so the approved view is viewable without AWS.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"os"
	"path/filepath"
)

// Job is one tile on the board. Field names are the /api/jobs contract the
// page's JS reads; the values map back to SPEC §9 job tags where noted.
type Job struct {
	ID            string    `json:"id"`                      // job name suffix, e.g. run-03
	Status        string    `json:"status"`                  // PENDING|IN_PROGRESS|COMPLETED|RESUMING
	Hypothesis    string    `json:"hypothesis"`              // tag: Hypothesis (head.tile_label)
	MetricName    string    `json:"metricName"`              // tag: Metric (bare, e.g. rmse)
	MetricValue   *float64  `json:"metricValue,omitempty"`   // latest value (CloudWatch / FinalMetricDataList)
	Domain        string    `json:"domain"`                  // tag: Domain (display only — board stays domain-blind)
	Instance      string    `json:"instance"`                // tag: Instance, e.g. g5.xlarge
	Spot          bool      `json:"spot"`                    // tag: Spot
	Step          int       `json:"step"`                    // progress
	TotalSteps    int       `json:"totalSteps"`              //
	Curve         []float64 `json:"curve"`                   // metric series for the sparkline
	ReclaimAt     *float64  `json:"reclaimAt,omitempty"`     // x (0..200) of a spot reclaim marker
	ResumedAtCkpt *int      `json:"resumedAtCkpt,omitempty"` // checkpoint step a RESUMING job rejoined from
}

type feed struct {
	Sweep string `json:"sweep"`
	Jobs  []Job  `json:"jobs"`
}

func f(v float64) *float64 { return &v }
func i(v int) *int         { return &v }

// sampleFeed mirrors the approved mockup's six tiles, shaped per §9. Replaced by
// live AWS reads in build-order (c).
func sampleFeed() feed {
	reclaim := f(100)
	return feed{
		Sweep: "mol-esol-sample",
		Jobs: []Job{
			{ID: "run-01", Status: "COMPLETED", Hypothesis: "shallow · feat ecfp", MetricName: "rmse", MetricValue: f(0.41), Domain: "molecular", Instance: "g5.xlarge", Step: 500, TotalSteps: 500, Curve: []float64{38, 30, 24, 18, 13, 9, 6, 5, 4}},
			{ID: "run-02", Status: "IN_PROGRESS", Hypothesis: "deep · feat ecfp", MetricName: "rmse", MetricValue: f(0.38), Domain: "molecular", Instance: "g5.xlarge", Step: 372, TotalSteps: 500, Curve: []float64{39, 32, 25, 21, 16, 13, 11}},
			{ID: "run-03", Status: "IN_PROGRESS", Hypothesis: "gnn · feat graph", MetricName: "rmse", MetricValue: f(0.31), Domain: "molecular", Instance: "g5.xlarge", Step: 368, TotalSteps: 500, Curve: []float64{37, 31, 27, 24, 22, 20, 19}},
			{ID: "run-04", Status: "IN_PROGRESS", Hypothesis: "gnn · feat graph+3d", MetricName: "rmse", MetricValue: f(0.39), Domain: "molecular", Instance: "g5.xlarge", Step: 360, TotalSteps: 500, Curve: []float64{38, 31, 24, 19, 15, 12, 10}},
			{ID: "run-05", Status: "RESUMING", Hypothesis: "deep · feat graph", MetricName: "rmse", MetricValue: f(0.34), Domain: "molecular", Instance: "g5.xlarge", Spot: true, Step: 300, TotalSteps: 500, Curve: []float64{37, 30, 25, 21, 18, 16, 15}, ReclaimAt: reclaim, ResumedAtCkpt: i(250)},
			{ID: "run-06", Status: "PENDING", Hypothesis: "gnn · feat graph+3d", MetricName: "rmse", Domain: "molecular", Instance: "g5.xlarge", Step: 0, TotalSteps: 500},
		},
	}
}

func main() {
	addr := flag.String("addr", "127.0.0.1:8080", "listen address (loopback for the Studio proxy)")
	tmpl := flag.String("template", "", "path to board_template.html (default: next to the binary / CWD)")
	sweep := flag.String("sweep", "", "SageMaker sweep id to scope live ListTrainingJobs (empty = sample feed)")
	ec2Sweep := flag.String("ec2-sweep", "", "EC2 self-managed-spot sweep id (reads instances by Sweep tag; stage-5 FIS demo)")
	region := flag.String("region", "", "AWS region for live reads (default: SDK chain)")
	flag.Parse()

	// Three modes, all read-only: --sweep (SageMaker jobs), --ec2-sweep (EC2
	// self-managed spot, for the FIS reclaim demo), or neither (sample feed so
	// the approved view renders with no AWS access).
	var live *liveClient
	var ec2c *ec2Client
	switch {
	case *ec2Sweep != "":
		c, err := newEC2Client(context.Background(), *region)
		if err != nil {
			log.Fatalf("ec2 client: %v", err)
		}
		ec2c = c
		log.Printf("ec2 mode: scoping spot sweep %q by tag (read-only)", *ec2Sweep)
	case *sweep != "":
		lc, err := newLiveClient(context.Background(), *region)
		if err != nil {
			log.Fatalf("live client: %v", err)
		}
		live = lc
		log.Printf("live mode: scoping SageMaker sweep %q (read-only)", *sweep)
	default:
		log.Printf("sample mode: no --sweep/--ec2-sweep given, serving sample feed")
	}

	templatePath := *tmpl
	if templatePath == "" {
		if _, err := os.Stat("board_template.html"); err == nil {
			templatePath = "board_template.html"
		} else if exe, err := os.Executable(); err == nil {
			templatePath = filepath.Join(filepath.Dir(exe), "board_template.html")
		}
	}

	mux := http.NewServeMux()

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		http.ServeFile(w, r, templatePath)
	})

	// Relative-fetched by the page as 'api/jobs' (proxy strips the prefix, §5.2).
	mux.HandleFunc("/api/jobs", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		out := sampleFeed()
		switch {
		case ec2c != nil:
			f, err := ec2c.fetchEC2Sweep(r.Context(), *ec2Sweep)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadGateway)
				return
			}
			out = f
		case live != nil:
			f, err := live.fetchSweep(r.Context(), *sweep)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadGateway)
				return
			}
			out = f
		}
		if err := json.NewEncoder(w).Encode(out); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})

	log.Printf("board serving on http://%s  (template: %s)", *addr, templatePath)
	srv := &http.Server{Addr: *addr, Handler: mux}
	log.Fatal(srv.ListenAndServe())
}

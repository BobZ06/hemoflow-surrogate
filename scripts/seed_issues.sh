#!/usr/bin/env bash
# Create the backlog on GitHub, labelled and assigned by lane.
#
# The issues below are the roadmap in `docs/ROADMAP.md`, one issue per item,
# split along the same boundary `CONTRIBUTING.md` draws so that two people can
# pick up work without reading each other's diffs. Run once, after the repo
# exists and both collaborators are added:
#
#     gh auth login
#     bash scripts/seed_issues.sh <owner>/<repo> <lane-a-user> <lane-b-user>
#
# Re-running creates duplicates; it is a seed, not a sync.
set -euo pipefail

REPO="${1:?usage: seed_issues.sh <owner>/<repo> <lane-a-user> <lane-b-user>}"
LANE_A="${2:?lane A github username (geometry, data)}"
LANE_B="${3:?lane B github username (models, serving)}"

command -v gh >/dev/null || { echo "gh CLI not found: https://cli.github.com" >&2; exit 1; }

label() {
  gh label create "$1" --repo "$REPO" --color "$2" --description "$3" 2>/dev/null || true
}

label "lane-a"       "0e8a16" "Geometry, reference solver, dataset generation"
label "lane-b"       "1d76db" "Architectures, training, metrics, service, demo"
label "roadmap"      "5319e7" "Ordered work from docs/ROADMAP.md"
label "correctness"  "b60205" "Silently produces plausible-looking wrong numbers"
label "good-first"   "c2e0c6" "Small, self-contained, well-specified"

issue() {
  local title="$1" assignee="$2" labels="$3" body="$4"
  gh issue create --repo "$REPO" --title "$title" --assignee "$assignee" \
    --label "$labels" --body "$body"
}

issue "Replace the reduced-order reference with real CFD targets" "$LANE_A" \
  "lane-a,roadmap" \
"The single highest-value change in the repo; everything else is scaffolding for it.

\`geometry/reference.py\` is a reduced-order model, not a 3D Navier-Stokes solve, so
every metric in the README measures fidelity to it rather than clinical accuracy.

The Vascular Model Repository (vascularmodel.com) ships geometries together with
simulation results and is the first source worth trying.

Scope:
- [ ] Loader that reads solver output into \`(n_arc, n_theta)\` arrays on our grid
- [ ] Swap the \`reference_wss(...)\` call in \`data/generate.py::generate_dataset\`
- [ ] Regenerate; the dataset hash changes, so old runs keep pointing at the data
      that produced them

Features, models, training, metrics and the service are untouched by this. See
\`docs/ARCHITECTURE.md\`, 'The reference solver, and how to replace it'."

issue "Segmented patient geometries instead of synthetic tubes" "$LANE_A" \
  "lane-a,roadmap" \
"CT/MR -> segmentation -> surface mesh -> centerline extraction -> resample onto the
\`(arc x circumference)\` grid. TotalSegmentator is the fast option for the first
step; VMTK for centerlines, with a distance-transform skeleton as the fallback.

Only the final resampling step touches this repo. The representation is already
the interface, which is the point of having built it that way.

Blocked by nothing, but lower value than real targets until targets are real."

issue "Bifurcations: the representation does not survive a branch" "$LANE_A" \
  "lane-a,roadmap" \
"An unwrapped \`(arc x circumference)\` grid assumes a single tube. Clinically
interesting lesions are disproportionately at branch points, where flow divides and
low-shear regions form on the outer walls.

This is a real representational change rather than a tweak, and it is where a graph
network would finally earn its complexity over a U-Net - see 'Deliberately not done'
in \`docs/ROADMAP.md\` for why we have not reached for one yet.

Design first, code second. Open a short design note before writing a model."

issue "Calibrated uncertainty on the predicted field" "$LANE_B" \
  "lane-b,roadmap,correctness" \
"A surrogate that cannot say when it is out of distribution should not be near a
clinical decision, and this one currently cannot.

The demo marks the training envelope on its sliders, which is the cheap version of
this and not a substitute for it.

Options, cheapest first:
- [ ] Quantile regression on the log-ratio target
- [ ] Deep ensemble (5 seeds), which also gives a free variance estimate

This matters more than another point of accuracy. Acceptance: the service returns a
per-node interval, and a test asserts coverage on the held-out split."

issue "Pulsatile flow and oscillatory shear index" "$LANE_B" \
  "lane-b,roadmap" \
"Steady flow gives one field. Real hemodynamics is time-resolved, and OSI - how much
shear direction reverses over a cardiac cycle - is arguably a better plaque predictor
than mean magnitude.

Needs time-series targets (so it is blocked behind real solver output) and a temporal
model. Large. Do not start this before uncertainty."

issue "Demo: batch a cohort instead of one vessel at a time" "$LANE_B" \
  "lane-b,good-first" \
"\`serving/bench.py\` already measures batch throughput separately from interactive
latency because they answer different questions. The demo only ever shows the
interactive one.

A second demo view that scores a few hundred vessels and ranks them by atheroprone
area fraction would show the other thing a surrogate buys you, which is cohort
screening rather than one interactive case.

Self-contained: new route in \`serving/demo.py\`, new panel in the page, no changes
to models or geometry."

issue "Pin the circumferential seam with a test on the trained model" "$LANE_B" \
  "lane-b,correctness,good-first" \
"\`MixedPadConv2d\` and \`topology_aware_upsample\` exist because zero-padding invents
a discontinuity at theta=0 that cuts through the Dean-flow variation. The padding
itself is tested; the property it exists to protect is not tested end to end.

Add a test that takes a trained U-Net, predicts a field, and asserts the magnitude of
the jump across the theta=0 seam is comparable to the jump between any two adjacent
circumferential stations. That is the assertion that would have caught the decoder
regression, which the padding tests did not."

issue "Feature ablation: which of the twelve features actually earn their place" "$LANE_B" \
  "lane-b,good-first" \
"The negative U-Net result is explained by the features already carrying the
non-local information - \`throat_ratio\` and \`downstream_of_throat\` in particular.
That explanation is currently an argument, not a measurement.

Drop one feature at a time, retrain the MLP, record relative L2. Twelve short runs.
Write the table into the README next to the model table.

Good first issue: no new concepts, and the result is genuinely interesting either way."

echo
echo "Seeded. Review at: https://github.com/$REPO/issues"

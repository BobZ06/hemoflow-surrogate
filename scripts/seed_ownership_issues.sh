#!/usr/bin/env bash
# Seed the current-sprint ownership split as GitHub issues.
#
# Distinct from seed_issues.sh: that script files the *roadmap* (new work,
# docs/ROADMAP.md). This one files the *ownership checklist* for what already
# exists - each issue is "understand, run, verify, and be able to explain
# this module", not "build something new". Both scripts can be run once, on a
# repo that already has both collaborators added.
#
#     gh auth login
#     bash scripts/seed_ownership_issues.sh <owner>/<repo> <bowen-user> <letian-user>
set -euo pipefail

REPO="${1:?usage: seed_ownership_issues.sh <owner>/<repo> <bowen-user> <letian-user>}"
BOWEN="${2:?Bowen's github username}"
LETIAN="${3:?Letian's github username}"

command -v gh >/dev/null || { echo "gh CLI not found: https://cli.github.com" >&2; exit 1; }

label() {
  gh label create "$1" --repo "$REPO" --color "$2" --description "$3" 2>/dev/null || true
}

label "owner:bowen"  "0e8a16" "Bowen is primary owner: run it, modify it, explain the result"
label "owner:letian" "1d76db" "Letian is primary owner: run it, modify it, explain the result"
label "verify"       "fbca04" "Understand and verify existing code, not new work"
label "submission"   "5319e7" "Video, registration, Devpost - not code"

# Every issue body opens with the same standard, so "primary owner" means the
# same thing everywhere: not "I read the file", but "I can run it, break it on
# purpose, and explain the result to someone who wasn't in the room."
STANDARD="**Primary owner** means: you can run this, modify it, and explain *why* it's built this way and what the result means - not that you've read the file once.

"

issue() {
  local title="$1" primary="$2" owner_label="$3" body="$4"
  gh issue create --repo "$REPO" --title "$title" --assignee "$primary" \
    --label "verify,$owner_label" --body "$STANDARD$body"
}

issue "Physics: WSS formula, units, scale relationships" "$BOWEN" "owner:bowen" \
"**Primary: Bowen.** Secondary: Letian checks how the model consumes this.

Files: \`src/hemoflow/models/physics.py\`, \`docs/ARCHITECTURE.md\` ('Why the network
predicts a ratio, not pascals').

- [ ] Can derive \`tau = 4*mu*Q / (pi*r^3)\` from first principles and explain why
      dimensionless features can't recover it (Reynolds carries Q/r, shear needs Q/r^3)
- [ ] Can explain the 2.5-micron-vessel bug this fixed, in your own words
- [ ] Ran \`pytest tests/test_models.py -k physics\` and read what it actually checks
- [ ] Picked one non-obvious input (e.g. a vessel 10x smaller than training) and
      hand-verified the Poiseuille scale factor matches \`poiseuille_field(...)\`

**Deliverable:** a short written explanation of the formula and the unit chain,
plus one worked numerical example, added to a comment or doc you can point a
judge at."

issue "Geometry: vessel representation and the 12 features" "$BOWEN" "owner:bowen" \
"**Primary: Bowen.** Secondary: Letian confirms how the model uses the feature tensor.

Files: \`src/hemoflow/geometry/vessel.py\`, \`geometry/features.py\`, \`docs/ARCHITECTURE.md\`
('Why the representation is what it is').

- [ ] Can explain parallel transport vs. a Frenet frame, and why the seam matters
- [ ] Can name all 12 entries in \`FEATURE_NAMES\` and what each one is for, without looking
- [ ] Ran \`tests/test_geometry.py::test_features_are_rotation_invariant\` and can
      explain what it would catch if it failed
- [ ] Traced one vessel by hand: given a radius profile, sketch what
      \`throat_ratio\` and \`downstream_of_throat\` should look like

**Deliverable:** a one-page input-format note (what a vessel is, what the feature
grid looks like) that Letian can build the serving layer against without asking you."

issue "Data: generation, leakage-safe splits, normalization" "$BOWEN" "owner:bowen" \
"**Primary: Bowen.** Secondary: Letian checks the model can actually read this data.

Files: \`src/hemoflow/data/generate.py\`, \`data/dataset.py\`, \`data/normalize.py\`.

- [ ] Can explain why splits are by whole vessel, not by surface node, and what
      would go wrong if they weren't
- [ ] Can explain why normalizers are fit on train only and persisted with the checkpoint
- [ ] Ran \`hemoflow data --config configs/mlp.yaml\` from scratch and read the manifest
- [ ] Deliberately broke something (e.g. shuffled a split) and confirmed
      \`test_data.py::test_splits_are_disjoint_by_vessel\` catches it, then reverted

**Deliverable:** confirm the dataset regenerates byte-for-byte reproducibly from a
config, and write down the one command that proves it."

issue "Models: MLP and U-Net implementation" "$LETIAN" "owner:letian" \
"**Primary: Letian.** Secondary: Bowen understands inputs/outputs and reviews.

Files: \`src/hemoflow/models/nets.py\`, \`models/base.py\`.

- [ ] Can explain \`MixedPadConv2d\` (circular in theta, replicate along arc) and
      the seam bug it fixes
- [ ] Can explain \`topology_aware_upsample\` and the decoder regression it fixes
- [ ] Ran both \`configs/mlp.yaml\` and \`configs/unet.yaml\` end to end
- [ ] Can justify, from the README table, why the U-Net's extra parameters buy nothing here

**Deliverable:** both architectures training and scoring cleanly from a clean
checkout - this is the 'can other people run my code' bar."

issue "Physics prior: the residual formulation and its ablation" "$BOWEN" "owner:bowen" \
"**Primary: Bowen** (designs the explanation and the verification). Secondary: Letian
(makes sure the model-side wiring and checkpoint loading respect the flag).

Files: \`src/hemoflow/models/physics.py\`, \`models/registry.py\`
(\`cfg.model.physics_residual\`), \`configs/mlp_no_physics.yaml\`.

- [ ] Trained both \`configs/mlp.yaml\` and \`configs/mlp_no_physics.yaml\`
- [ ] Can state the relative-L2 gap from memory and explain it without notes
- [ ] Confirmed in the demo that toggling the switch actually loads the other
      checkpoint (not just relabeling the same predictions)

**Deliverable:** the with/without comparison, reproducible from two commands,
matching the README table."

issue "Training: experiment configuration, runs, and records" "$BOWEN" "owner:bowen" \
"**Primary: Bowen** (config, running, recording results). Secondary: Letian handles
model-side implementation issues that come up mid-run.

Files: \`src/hemoflow/training/train.py\`, \`configs/*.yaml\`.

- [ ] Can explain early stopping and best-checkpoint selection in this codebase
- [ ] Ran the full training matrix (mlp, mlp_no_physics, unet) and has the run
      directories to show for it
- [ ] Can explain \`Config.run_id\` vs. \`Config.dataset_id\` and why they hash different fields

**Deliverable:** a reproducible experiment log - which config produced which run
directory, and the command that reproduces each one."

issue "Evaluation: baselines, error distribution, failure cases" "$BOWEN" "owner:bowen" \
"**Primary: Bowen** (baselines, error distribution, failure cases). Secondary: Letian
independently checks the evaluation method itself.

Files: \`src/hemoflow/training/evaluate.py\`, \`training/metrics.py\`.

- [ ] Can explain why relative L2 alone is a bad primary metric here (two orders
      of magnitude span, dominated by stenosis-throat nodes)
- [ ] Can explain the low-shear Dice metric and why it's the one closest to the
      actual clinical question
- [ ] Found at least one vessel where the surrogate does worst, and can say why

**Deliverable:** the benchmark table reproduced from a clean run, plus one
concrete failure case with an explanation."

issue "Checkpointing: save, load, and reload consistency" "$LETIAN" "owner:letian" \
"**Primary: Letian** (weights, config, normalizer state, loading behavior).
Secondary: Bowen verifies results are identical before and after a save/reload.

Files: \`src/hemoflow/models/registry.py\` (\`save_checkpoint\`, \`load_surrogate\`).

- [ ] Can explain why a run directory is self-contained (weights + standardizer +
      target transform + config all together)
- [ ] Ran \`test_models.py::test_checkpoint_roundtrip_preserves_predictions\` and
      can explain what it would miss if normalizer state weren't saved
- [ ] Manually reloaded a checkpoint in a fresh Python process and confirmed
      identical predictions to the original training run

**Deliverable:** a checkpoint that reloads cleanly on a different machine
(Bowen's, if it was trained on Letian's, or vice versa)."

issue "Serving: API, demo, and error handling" "$LETIAN" "owner:letian" \
"**Primary: Letian** (interface, page, error handling). Secondary: Bowen checks
the numbers, units, and conclusions the demo presents are correct.

Files: \`src/hemoflow/serving/api.py\`, \`serving/demo.py\`, \`serving/static/demo.html\`.

- [ ] Can explain why /predict and the demo are separate modules
- [ ] Can explain the physics-prior toggle end to end: request -> which checkpoint
      loads -> what changes in the response
- [ ] Tried at least 3 invalid inputs against /predict and confirmed each fails
      with a clear 422, not a wrong-but-confident answer
- [ ] Ran the full demo with wifi off and confirmed it still works

**Deliverable:** the demo, cold-started, on both of your laptops independently."

issue "Submission: video, registration, Devpost" "$BOWEN" "submission" \
"Shared, but Bowen leans on the experiment/limitations framing and Letian leans on
recording and mechanics. See the judging call sheet artifact for the full script,
Q&A prep, and Devpost copy - this issue just tracks that the mechanical steps got done.

- [ ] 3-minute video recorded and uploaded (unlisted YouTube is fine)
- [ ] Devpost project created, both members added
- [ ] Track selected: Toralis Labs Healthcare only
- [ ] Team registration form submitted (counts toward school points)
- [ ] Posted in Discord showcase channel
- [ ] Both of you drilled the Q&A out loud at least once before judging"

echo
echo "Seeded. Review at: https://github.com/$REPO/issues"

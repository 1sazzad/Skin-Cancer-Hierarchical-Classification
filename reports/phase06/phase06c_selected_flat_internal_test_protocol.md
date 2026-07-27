# Phase 06C Selected Flat Internal-Test Protocol

## Frozen purpose and identity

This protocol authorizes a one-time internal-test evaluation of the
validation-selected flat four-class EfficientNet-B0. Validation-only model
selection froze Phase 06A clean cross-entropy, seed `42`, before internal-test
access.

- Selected checkpoint:
  `runs/phase06_full/full__phase06_flat_four_class_isic2019_efficientnet_b0_cross_entropy_seed42__20260726T232308Z/best_checkpoint.pt`
- SHA-256:
  `f3d8b8b0e5ef42e3c287a2377b5570411d442246acd16cb874ccf903facdc7a7`
- Allowed checkpoint count: `1`
- Allowed candidate count: `1`
- Manifest split value: `test` (the evaluator exposes this as loader key
  `internal_test`)
- Prepared state: `internal_test_accessed=false`
- Final state: `internal_test_accessed=true`, `protocol_status=consumed_locked`

The selected checkpoint is frozen before internal-test access. There is no
validation re-selection, hyperparameter tuning, threshold tuning after test
access, focal-candidate evaluation, post-test tuning, or candidate switching.
The flat result must be locked before comparison with the already locked
hierarchy result. A second valid run merely to improve metrics is forbidden.

Only a technical retry is allowed when the preceding attempt failed before
producing valid internal-test metrics. Its failure reason and artifacts must be
preserved. Once valid metrics exist, this protocol is consumed and the result
is locked. Model execution is forbidden locally and must occur on the Azure
Tesla T4 VM.

## Historical Azure Tesla T4 command — consumed; do not rerun

The executable section contains only the selected clean-CE checkpoint. It
was executed exactly once on the Azure Tesla T4 VM and must not be executed
again.

```bash
source "$HOME/venvs/skin-cancer-gpu/bin/activate"
export REPO="$HOME/projects/Skin-Cancer-Hierarchical-Classification"
export BRANCH="phase06c-selected-flat-internal-test"
export CHECKPOINT="runs/phase06_full/full__phase06_flat_four_class_isic2019_efficientnet_b0_cross_entropy_seed42__20260726T232308Z/best_checkpoint.pt"
export EXPECTED_SHA256="f3d8b8b0e5ef42e3c287a2377b5570411d442246acd16cb874ccf903facdc7a7"
export PROTOCOL="configs/evaluation/phase06c_selected_flat_internal_test.yaml"
export OUTPUT="runs/phase06c/selected_flat_internal_test/locked_primary_evaluation"
export CONTROL="runs/phase06c/selected_flat_internal_test/control"
cd "$REPO"
git fetch origin
git switch "$BRANCH"
git pull --ff-only origin "$BRANCH"
test -z "$(git status --porcelain)"
export EXPECTED_COMMIT="$(git rev-parse "origin/$BRANCH")"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
git show -s --format='%H %s' "$EXPECTED_COMMIT"
python --version
python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.get_device_name(0) == 'Tesla T4'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
nvidia-smi
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$EXPECTED_SHA256"
python -c "import yaml; p=yaml.safe_load(open('$PROTOCOL')); s=p['selected_model']; assert p['protocol_status']=='prepared_not_executed'; assert not p['internal_test_accessed']; assert p['allowed_checkpoint_count']==p['allowed_candidate_count']==1; assert p['internal_test_split']=='test'; assert s['phase']=='06A' and s['loss']=='cross_entropy'; assert s['checkpoint_path']=='$CHECKPOINT'; assert s['checkpoint_sha256']=='$EXPECTED_SHA256'; assert not p['local_model_execution_allowed']"
mkdir -p "$CONTROL"
cp "$PROTOCOL" "$CONTROL/resolved_evaluation_protocol.yaml"
git rev-parse HEAD > "$CONTROL/git_commit.txt"
sha256sum "$CHECKPOINT" > "$CONTROL/selected_checkpoint.sha256"
date -u +%Y-%m-%dT%H:%M:%SZ > "$CONTROL/evaluation_started_at_utc.txt"
tmux new-session -d -s phase06c_selected_flat_test "cd '$REPO' && source \"\$HOME/venvs/skin-cancer-gpu/bin/activate\" && set -o pipefail; PYTHONUNBUFFERED=1 python -u scripts/evaluate_isic2019_internal_test.py --checkpoint '$CHECKPOINT' --project-root '$REPO' --output-directory '$OUTPUT' --device cuda 2>&1 | tee '$CONTROL/evaluation.log'; code=\${PIPESTATUS[0]}; printf '%s\n' \"\$code\" > '$CONTROL/final_status.txt'; exit \"\$code\""
```

The final `tmux new-session` command started the only authorized internal-test
evaluation. Valid metrics were produced with exit status `0`; therefore the
one-time protocol is consumed. Do not alter or rerun the evaluator.

Historical completion-validation commands:

```bash
cd "$REPO"
tmux has-session -t phase06c_selected_flat_test
tmux capture-pane -pt phase06c_selected_flat_test -S -200
ps -ef | grep '[e]valuate_isic2019_internal_test.py'
nvidia-smi
tail -n 100 runs/phase06c/selected_flat_internal_test/control/evaluation.log
cat runs/phase06c/selected_flat_internal_test/control/final_status.txt
test "$(cat runs/phase06c/selected_flat_internal_test/control/final_status.txt)" = "0"
test -s runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/internal_test_metrics.json
test -s runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/internal_test_predictions.csv
test -s runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/confusion_matrix.csv
test -s runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/per_class_metrics.csv
test -s runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/evaluation_summary.json
python -m json.tool runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/evaluation_summary.json
export ARCHIVE="phase06c_selected_flat_internal_test_$(git rev-parse --short=12 HEAD).tar.gz"
tar -czf "$ARCHIVE" runs/phase06c/selected_flat_internal_test
sha256sum "$ARCHIVE" | tee "$ARCHIVE.sha256"
```

Only after those required artifacts contain valid metrics may the repository
metadata be changed to `internal_test_accessed=true`,
`valid_internal_test_run_completed=true`, and a consumed/locked status.

Copy the archive back from the local source-of-truth PC (the command prompts
for the VM SSH target so no machine identity is guessed):

```powershell
cd "F:\Research\Final Year\Skin-Cancer-Hierarchical-Classification"
$Vm = Read-Host "Azure VM SSH target (user@host)"
$RemoteRepo = Read-Host "Absolute Azure repository path"
$Archive = Read-Host "Archive filename printed by the VM command"
scp "${Vm}:${RemoteRepo}/${Archive}" "runs\backups\phase06c\"
scp "${Vm}:${RemoteRepo}/${Archive}.sha256" "runs\backups\phase06c\"
Get-FileHash -Algorithm SHA256 -LiteralPath "runs\backups\phase06c\$Archive"
Get-Content -LiteralPath "runs\backups\phase06c\$Archive.sha256"
```

The computed local SHA-256 must exactly match the transferred hash record.


## Executed outcome — consumed and locked

The one authorized evaluation completed successfully on the Azure Tesla T4.

- Evaluation commit: `550e7cdb1144f059c940d4240fe4579e0280a803`
- Started at: `2026-07-27T15:22:59Z`
- Completed at: `2026-07-27T15:34:03Z`
- Final status: `0`
- Checkpoint epoch: `2`
- Internal-test samples: `3668`
- Accuracy: `0.7420937841`
- Balanced accuracy: `0.6503125394`
- Macro-F1: `0.6192224685`
- Weighted F1: `0.7525567214`
- Mean loss: `0.6232672186`
- Metrics: `runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/internal_test_metrics.json`
- Predictions: `runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/internal_test_predictions.csv`
- Local verified archive: `runs/backups/phase06c/phase06c_selected_flat_internal_test_550e7cdb1144.tar.gz`
- Archive SHA-256:
  `b76762b53a35a8d9b0aa96621d78ea0e4421aa6e8052d068ffc10648a4e63e91`
- Embedded artifact manifest entries: `12`

The locked Phase 05 predicted-gate hierarchical macro-F1 was
`0.6053674006`. The selected flat model is higher by
`0.0138550680` on this single locked internal split. This is a
descriptive comparison only; it does not establish statistical significance,
clinical superiority, fairness, or external generalisation.

The protocol state is now `consumed_locked`,
`internal_test_accessed=true`, and
`valid_internal_test_run_completed=true`. No technical retry or performance
rerun is permitted because valid metrics already exist.

## Claims boundary

This protocol produces one locked internal estimate. It cannot support a claim
of clinical readiness, fairness, external generalisation, state-of-the-art
performance, statistical significance, or flat-model superiority by itself.

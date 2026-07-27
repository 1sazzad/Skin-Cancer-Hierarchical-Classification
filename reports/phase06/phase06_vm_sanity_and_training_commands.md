# Phase 06B Azure Tesla T4 Commands

Do not run these commands on the local PC. Replace only the explicit
placeholders. Every full training job runs inside `tmux`.

## Activate, synchronize, and verify

```bash
source "$HOME/venvs/skin-cancer-gpu/bin/activate"
export REPO="$HOME/Skin-Cancer-Hierarchical-Classification"
cd "$REPO"
git fetch origin
git switch phase06b-flat-four-class-cb-focal
git pull --ff-only origin phase06b-flat-four-class-cb-focal
git rev-parse HEAD
git status --short --branch
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
nvidia-smi
```

Optionally remove only the known redundant clean-CE archive when the VM is
running. Never remove the verified archive:

```bash
test -f backups/phase06/phase06_clean_ce_training_backup_20260726T234503Z.tar.gz &&
  rm -- backups/phase06/phase06_clean_ce_training_backup_20260726T234503Z.tar.gz
test -f backups/phase06/phase06_clean_ce_training_backup_verified_20260726T235849Z.tar.gz
```

## Full VM tests and label audit

```bash
cd "$REPO"
python -m pytest -q
python scripts/audit_phase06_flat_labels.py
```

## Non-reportable CUDA sanity

```bash
cd "$REPO"
mkdir -p runs/phase06b/sanity
PYTHONUNBUFFERED=1 python -u scripts/train_isic2019_baseline.py \
  --config configs/experiments/phase06b_flat_four_class_isic2019_efficientnet_b0_class_balanced_focal_loss.yaml \
  --project-root "$REPO" \
  --output-root runs/phase06b/sanity \
  --device cuda \
  --max-train-batches 1 \
  --max-validation-batches 1 \
  --epoch-limit 1 2>&1 | tee runs/phase06b/sanity/cuda_sanity.log
```

Verify Tesla T4; task `flat_four_class`; output dimension four; exact class
order; class-balanced focal loss; beta `0.9999`; gamma `2.0`; counts
`11193/3164/2327/440`; ordered weights with SCC largest; checkpoints, resolved
config, and run summary present; `sanity_run: true`;
`reportable_as_full_result: false`; and no internal-test metrics.

```bash
find runs/phase06b/sanity -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
tail -n 100 runs/phase06b/sanity/cuda_sanity.log
grep -R -nE 'sanity_run|reportable_as_full_result|class_weights|focal_gamma' runs/phase06b/sanity
```

## Full training in tmux

```bash
cd "$REPO"
mkdir -p runs/phase06b/full
tmux new-session -d -s phase06b_cb_focal \
  "cd '$REPO' && source \"\$HOME/venvs/skin-cancer-gpu/bin/activate\" && \
   set -o pipefail; PYTHONUNBUFFERED=1 python -u scripts/train_isic2019_baseline.py \
   --config configs/experiments/phase06b_flat_four_class_isic2019_efficientnet_b0_class_balanced_focal_loss.yaml \
   --project-root '$REPO' --output-root runs/phase06b/full --device cuda \
   2>&1 | tee runs/phase06b/full/training.log; \
   code=\${PIPESTATUS[0]}; printf '%s\n' \"\$code\" > runs/phase06b/full/final_status.txt; exit \"\$code\""
```

Monitor without treating session existence as proof of success:

```bash
tmux has-session -t phase06b_cb_focal
tmux capture-pane -pt phase06b_cb_focal -S -200
ps -ef | grep '[t]rain_isic2019_baseline.py'
nvidia-smi
tail -n 100 runs/phase06b/full/training.log
find runs/phase06b/full -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
cat runs/phase06b/full/final_status.txt
python -m json.tool runs/phase06b/full/full__EXACT_RUN_DIRECTORY/run_summary.json
python -m json.tool runs/phase06b/full/full__EXACT_RUN_DIRECTORY/best_validation_metrics.json
```

Require final status `0`, a complete run directory, and validation artifacts.
Do not run internal-test evaluation. Compare Phase 06A and Phase 06B using
validation macro-F1, then validation balanced accuracy; retain clean CE if both
are exactly tied.

## Hash, archive, and transfer

```bash
cd "$REPO"
export RUN_DIR="runs/phase06b/full/full__EXACT_RUN_DIRECTORY"
export ARCHIVE="phase06b_cb_focal_EXACT_COMMIT.tar.gz"
sha256sum "$RUN_DIR/best_checkpoint.pt" | tee "$RUN_DIR/best_checkpoint.sha256"
git rev-parse HEAD > "$RUN_DIR/git_commit.txt"
tar -czf "$ARCHIVE" "$RUN_DIR" \
  reports/phase06/phase06b_class_balanced_focal_amendment.md \
  configs/experiments/phase06b_flat_four_class_isic2019_efficientnet_b0_class_balanced_focal_loss.yaml
sha256sum "$ARCHIVE" | tee "$ARCHIVE.sha256"
scp "$ARCHIVE" "$ARCHIVE.sha256" \
  USER@LOCAL_HOST:'F:/Research/Final Year/Skin-Cancer-Hierarchical-Classification/runs/backups/phase06/'
```

On the local PC, preserve the archive in the source-of-truth folder and verify:

```powershell
$Backup = 'F:\Research\Final Year\Skin-Cancer-Hierarchical-Classification\runs\backups\phase06'
$Archive = Join-Path $Backup 'phase06b_cb_focal_EXACT_COMMIT.tar.gz'
Get-FileHash -Algorithm SHA256 -LiteralPath $Archive
$Verify = Join-Path $env:TEMP 'phase06b_cb_focal_verify'
New-Item -ItemType Directory -Path $Verify
tar -xzf $Archive -C $Verify
Get-ChildItem -Recurse -File -LiteralPath $Verify
Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $Verify 'runs\phase06b\full\full__EXACT_RUN_DIRECTORY\best_checkpoint.pt')
```

Match the local archive hash to the VM `.sha256`, verify required files and the
extracted checkpoint hash, and retain the archive. Do not delete the VM run
until local verification succeeds.

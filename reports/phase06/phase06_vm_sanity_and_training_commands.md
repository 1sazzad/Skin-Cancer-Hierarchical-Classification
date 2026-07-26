# Phase 06 Azure T4 Commands

Set the repository and environment placeholders once:

```bash
export REPO=/path/to/Skin-Cancer-Hierarchical-Classification
export ENV_ACTIVATE=/path/to/venv/bin/activate
cd "$REPO"
```

## Synchronize and preflight

```bash
git fetch origin
git switch phase06-flat-four-class-baseline
git pull --ff-only origin phase06-flat-four-class-baseline
source "$ENV_ACTIVATE"
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
test -f data/manifests/isic2019_train_val_test_split_seed42.csv
python scripts/audit_phase06_flat_labels.py
pytest -q tests/test_phase06_flat_four_class.py tests/test_isic2019_dataset.py tests/test_dataloaders.py tests/test_phase03_baseline_experiment.py
pytest -q
```

Confirm the dataset image root exists, then run the config’s static path audit:

```bash
python -c "from pathlib import Path; import pandas as pd; f=pd.read_csv('data/manifests/isic2019_train_val_test_split_seed42.csv'); p=[Path(x) for x in f.loc[(f.split_included==1)&(f.include_stage_1==1),'image_path']]; missing=[str(x) for x in p if not x.is_file()]; print({'eligible':len(p),'missing':len(missing),'examples':missing[:3]}); raise SystemExit(bool(missing))"
```

## Non-reportable CUDA sanity

This is the exact first model-execution command. It uses one train batch, one
validation batch, one epoch, and never iterates the internal-test loader.

```bash
mkdir -p runs/phase06
python scripts/train_isic2019_baseline.py \
  --config configs/experiments/phase06_flat_four_class_isic2019_efficientnet_b0_cross_entropy.yaml \
  --project-root "$REPO" \
  --output-root runs/phase06 \
  --device cuda \
  --max-train-batches 1 \
  --max-validation-batches 1 \
  --epoch-limit 1 2>&1 | tee runs/phase06/cuda_sanity.log
```

Inspect but do not promote sanity artifacts:

```bash
find runs/phase06 -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
tail -n 100 runs/phase06/cuda_sanity.log
nvidia-smi
```

If the sanity run is confirmed failed/incomplete, resolve the exact directory
from the log and remove only that directory:

```bash
find runs/phase06 -maxdepth 1 -type d -name 'sanity__*' -print
rm -rf -- runs/phase06/sanity__EXACT_FAILED_DIRECTORY
```

## Full training in tmux

Run only after sanity artifacts and logs pass review:

```bash
tmux new-session -s phase06-flat-ce
source "$ENV_ACTIVATE"
cd "$REPO"
python scripts/train_isic2019_baseline.py \
  --config configs/experiments/phase06_flat_four_class_isic2019_efficientnet_b0_cross_entropy.yaml \
  --project-root "$REPO" \
  --output-root runs/phase06 \
  --device cuda 2>&1 | tee runs/phase06/full_training.log
```

Detach with `Ctrl-b d`; inspect with:

```bash
tmux attach -t phase06-flat-ce
tmux capture-pane -pt phase06-flat-ce -S -200
tail -f runs/phase06/full_training.log
watch -n 2 nvidia-smi
ps -ef | grep '[t]rain_isic2019_baseline.py'
```

## Validation-only review

```bash
find runs/phase06 -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
python -m json.tool runs/phase06/full__EXACT_RUN_DIRECTORY/run_summary.json
python -m json.tool runs/phase06/full__EXACT_RUN_DIRECTORY/best_validation_metrics.json
sha256sum runs/phase06/full__EXACT_RUN_DIRECTORY/best_checkpoint.pt
```

Verify `reportable_as_full_result=true`, four semantic class names, selection
metric `macro_f1`, and a best checkpoint before freezing it. Do not inspect
internal-test predictions or metrics during this review.

## Future locked internal-test evaluation

Do not run this section until the selected checkpoint, Git commit, config, and
one-time evaluation authorization are frozen. Use a new output directory and
execute exactly once:

```bash
python scripts/evaluate_isic2019_internal_test.py \
  --checkpoint runs/phase06/full__EXACT_RUN_DIRECTORY/best_checkpoint.pt \
  --project-root "$REPO" \
  --output-directory runs/phase06/locked_internal_test \
  --device cuda
```

## Archive and return artifacts

```bash
git rev-parse HEAD > runs/phase06/git_commit.txt
sha256sum runs/phase06/full__EXACT_RUN_DIRECTORY/* > runs/phase06/sha256sums.txt
tar -czf phase06_flat_ce_EXACT_COMMIT.tar.gz runs/phase06 reports/phase06
sha256sum phase06_flat_ce_EXACT_COMMIT.tar.gz
scp phase06_flat_ce_EXACT_COMMIT.tar.gz USER@LOCAL_HOST:/path/to/local/source-of-truth/
```

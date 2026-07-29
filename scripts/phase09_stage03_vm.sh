#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$PWD}"
CONFIG="configs/experiments/phase09_stage03_emb_efficientnet_b0_cross_entropy.yaml"

case "${1:-help}" in
  update)
    git fetch origin phase09-stage03-fasttrack
    git switch phase09-stage03-fasttrack
    git pull --ff-only origin phase09-stage03-fasttrack
    ;;
  verify)
    git rev-parse HEAD
    python --version
    nvidia-smi
    python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
    ;;
  acquire)
    PROJECT_ROOT="$ROOT" bash scripts/acquire_emb_vm.sh
    ;;
  audit)
    python scripts/audit_emb_stage03.py --project-root "$ROOT"
    ;;
  split)
    python scripts/build_emb_stage03_split.py --project-root "$ROOT" \
      --input reports/dataset_audits/emb_stage03_available_images.csv
    ;;
  sanity)
    python scripts/train_isic2019_baseline.py --config "$CONFIG" \
      --project-root "$ROOT" --output-root experiments/runs --device cuda \
      --max-train-batches 2 --max-validation-batches 2 --epoch-limit 1
    ;;
  train)
    python scripts/train_isic2019_baseline.py --config "$CONFIG" \
      --project-root "$ROOT" --output-root experiments/runs --device cuda
    ;;
  evaluate)
    : "${CHECKPOINT:?Set CHECKPOINT to the full run best_checkpoint.pt}"
    : "${EVAL_OUTPUT:?Set EVAL_OUTPUT to a new evaluation directory}"
    python scripts/evaluate_isic2019_internal_test.py --checkpoint "$CHECKPOINT" \
      --project-root "$ROOT" --output-directory "$EVAL_OUTPUT" --device cuda
    ;;
  backup)
    : "${RUN_DIR:?Set RUN_DIR to the completed run directory}"
    tar --exclude='last_checkpoint.pt' -czf "${RUN_DIR%/}_compact.tgz" \
      "$RUN_DIR/resolved_config.yaml" "$RUN_DIR/environment.json" \
      "$RUN_DIR/history.csv" "$RUN_DIR/history.json" "$RUN_DIR/run_summary.json" \
      "$RUN_DIR/best_validation_metrics.json" "$RUN_DIR/best_checkpoint.pt"
    ;;
  *) echo "usage: $0 {update|verify|acquire|audit|split|sanity|train|evaluate|backup}";;
esac

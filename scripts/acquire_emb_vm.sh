#!/usr/bin/env bash
set -euo pipefail

# VM ONLY. This script deliberately refuses to run without CUDA-capable VM context.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_DIR="$PROJECT_ROOT/data/external/emb_official"
RAW_DIR="$PROJECT_ROOT/data/raw/emb"
EVIDENCE_DIR="$RAW_DIR/source_evidence"
REPO_URL="https://github.com/Oichii/EMB.git"

if [[ "$(uname -s)" != "Linux" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NO-GO: EMB acquisition is restricted to the Linux Azure GPU VM." >&2
  exit 20
fi
mkdir -p "$PROJECT_ROOT/data/external" "$RAW_DIR" "$EVIDENCE_DIR"
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git clone --filter=blob:none "$REPO_URL" "$SOURCE_DIR"
fi
git -C "$SOURCE_DIR" fetch --prune origin
git -C "$SOURCE_DIR" checkout --detach origin/main
git -C "$SOURCE_DIR" rev-parse HEAD | tee "$EVIDENCE_DIR/emb_git_commit.txt"
cp "$SOURCE_DIR/README.md" "$EVIDENCE_DIR/README.md"
cp "$SOURCE_DIR/early_melanoma_benchmark_dataset_labels.csv" "$RAW_DIR/"
find "$SOURCE_DIR" -maxdepth 2 -type f \
  \( -iname 'LICENSE*' -o -iname 'LICENCE*' -o -iname 'COPYING*' \) \
  -print | tee "$EVIDENCE_DIR/licence_files.txt"

if [[ ! -s "$EVIDENCE_DIR/licence_files.txt" ]]; then
  cat >&2 <<EOF
PROVENANCE FINDING: the official EMB repository does not provide an identifiable
licence. This command records source evidence only. It does not authorize or
download EMB or Dermoscopy Atlas images. Use the independent official ISIC API
workflow for per-image metadata, licence validation, and downloads.
EOF
  exit 0
fi

cat <<EOF
PROVENANCE FINDING: licence files were recorded for manual review.
This command does not download images. Dermoscopy Atlas remains excluded.
EOF

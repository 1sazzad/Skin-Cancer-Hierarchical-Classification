#!/usr/bin/env bash
set -euo pipefail

# VM ONLY. This script deliberately refuses to run without CUDA-capable VM context.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_DIR="$PROJECT_ROOT/data/external/emb_official"
RAW_DIR="$PROJECT_ROOT/data/raw/emb"
EVIDENCE_DIR="$RAW_DIR/source_evidence"
APPROVAL_FILE="$RAW_DIR/USAGE_TERMS_APPROVED.txt"
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

if [[ ! -s "$EVIDENCE_DIR/licence_files.txt" && ! -s "$APPROVAL_FILE" ]]; then
  cat >&2 <<EOF
NO-GO: the official EMB repository does not provide an identifiable licence.
Establish dataset/image usage permission with the owners and underlying ISIC/Atlas
sources. Record the approval, scope, date, and approver in:
  $APPROVAL_FILE
Then rerun this script. Do not download images until this gate is satisfied.
EOF
  exit 21
fi

cat <<EOF
GO FOR VM-ONLY ACQUISITION.
Follow the official README exactly:
  1. Export ISIC images from https://gallery.isic-archive.com/ with every
     Melanoma Thickness (mm) option and Melanoma Class='in situ'.
  2. Place exported images under: $RAW_DIR/images/isic/
  3. Review the official Atlas scraper in $SOURCE_DIR/wed_scraping.py, confirm
     dermoscopyatlas.com permission/terms, then place its images under:
     $RAW_DIR/images/atlas/
Never commit raw images or archives. Re-run the Stage-3 audit after acquisition.
EOF

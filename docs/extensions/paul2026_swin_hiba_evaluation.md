# PAUL Swin-T frozen HIBA evaluation

The implementation is `src/evaluation/paul2026_swin_hiba_evaluation.py`.
It imports the historical `scripts/run_phase06f_hiba_external.py` dataset,
prediction collectors, canonical manifest hashing, and patient-cluster bootstrap.
It uses the same checkpoint inspection and strict loading as the completed Swin
ISIC evaluator. No ISIC result or historical paper artifact is rewritten.

The frozen configuration is read from
`configs/paper/evaluation/phase06f_hiba_external.yaml`. The final manifest is used
directly, without new diagnosis filtering: 1,232 images, 568 patients, and class
counts 696 non-malignant, 196 melanoma, 229 BCC, 111 SCC. Its frozen canonical
CRLF SHA-256 is
`a2f30f14a249d8acb2bd9f03884e5e41c773ddbee19910d5b739407521e3706a`.
Preflight validates every image hash and the label/index and patient mapping.

Historical preprocessing is RGB conversion, bilinear antialiased resize of the
short side to 256, center crop to 224, float32 scaling, and ImageNet normalization.
The loader retains batch size 64, four workers, no shuffle, pinned memory,
persistent workers, prefetch factor 2, and generator seed 42.

Statistics call the historical patient-cluster function: sorted unique patients,
resample patients with replacement, retain all images for every sampled patient
(including repeated clusters), 5,000 replicates, NumPy generator seed 42, and
2.5th/97.5th percentile macro-F1 difference bounds. Both Hard minus Flat and
Oracle minus Hard use this function. The historical evaluator only reported the
first comparison and contains no McNemar test; this extension applies its same
paired bootstrap to the requested second comparison without adding other tests.
Oracle uses the true Task-1 gate with the same Shared-Hard checkpoint and Task-2
predictions. All three seeds are reported independently, then summarized using
arithmetic means and sample SD (`ddof=1`).

## Modal inputs and manual execution

The existing `paul2026-swin-data` volume is mounted at both `data/raw` and
`data/external/hiba/extracted`. HIBA images must therefore be available as
`images/ISIC_*.jpg` at that volume's root. The frozen manifest stays in the
deployed repository, outside the volume mount. Preflight fails if any required
image is missing or has a different hash. No alternate image location or cohort
is substituted. Image availability in the remote volume has not been checked.

Deploy the updated `scripts/modal_paul2026_swin.py` manually, then use the existing
evaluation launcher with `--mode hiba-preflight`, followed only after successful
preflight by `--mode hiba`. These modes spawn
`run_hiba_evaluation_preflight` (CPU only) and `run_hiba_evaluation_production`
(T4). Both use one maximum container, zero retries, single-use containers, and
the existing results volume. The launcher exits immediately after spawning.
Existing `preflight` and `isic` modes remain available.

## Output and recovery

All runtime output is below
`results/extensions/paul2026_swin/evaluation/hiba_external/`:

- `preflight_hiba_lock.json`: all six checkpoints/summaries, source/config/input
  hashes, the ordered frozen cohort, loader, preprocessing, and protocol identity.
- `seed42/`, `seed123/`, `seed2026/`: `metrics_and_statistics.json`,
  `paired_hiba_predictions.csv`, and `seed_complete.json`.
- `three_seed_summary.json` and `evaluation_complete.json`.

Production requires and revalidates the lock before any inference. Each seed is
published via an atomic staging-directory rename. Hidden staging directories
from an interrupted write are ignored, allowing a retry. Visible incomplete or
tampered seed directories fail closed; they are never silently skipped or
overwritten. Completed seeds are validated before any missing seed executes.
A fully completed evaluation returns its validated summary without inference.
The lock cannot be recreated after outputs exist. Changing locked source or
inputs requires investigating the drift rather than replacing the lock.

Regression tests are in `tests/test_paul2026_swin_hiba_evaluation.py`, with launcher
coverage extended in `tests/test_paul2026_swin_evaluation.py`. Tests, Python,
training, evaluations, Git, and Modal were not executed during implementation.

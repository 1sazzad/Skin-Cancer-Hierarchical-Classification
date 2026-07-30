# Complete Scientific Audit of Skin-Cancer-Hierarchical-Classification

**Audit date:** 2026-07-30
**Audit scope:** Repository-resident evidence only
**Intended use:** Primary technical reference for an ICCIT 2026 manuscript
**Audit mode:** Read-only scientific and software audit; no training, inference, code modification, deletion, commit, or push was performed

## 1. Audit conventions and evidence policy

This report uses three explicit evidence labels:

- **FACT** Ã¢â‚¬â€ directly supported by a named repository artifact.
- **INFERENCE** Ã¢â‚¬â€ a reasoned interpretation of one or more facts. It is not a directly recorded result.
- **RECOMMENDATION** Ã¢â‚¬â€ proposed future work or manuscript action. It is not completed evidence.

Numerical results are transcribed from repository metrics, reports, or generated
tables. No missing quantity has been estimated. Ã¢â‚¬Å“UnavailableÃ¢â‚¬Â means that no
repository-resident measurement was found.

The repository contains 34,704 non-Git paths reported by `rg --files -uu`,
including 34,134 paths under `.venv`. Excluding `.git` and `.venv`, the audit
found approximately 570 first-party/runtime artifacts. Git reports 260 tracked
files and the complete `paper/` directory as untracked. The first-party set
contains source, tests, configurations, reports, manifests, run outputs,
checkpoints, archives, an embedded Python runtime, and the manuscript. Binary
checkpoints and archives were inventoried by path, size, and existing audit
metadata; their tensor contents were not independently deserialized during this
audit.

Line references are supplied where they materially improve traceability. For
structured YAML/JSON/CSV artifacts, the field or row is often more stable and
precise than a line reference.

## 2. Executive scientific verdict

### 2.1 What the repository actually demonstrates

**FACT.** The completed primary experiment is a two-stage conditional
four-class classifier, not a completed three-stage clinical hierarchy. Stage 1
classifies `non_malignant` versus `malignant`; predicted malignant samples are
sent to Stage 2, which classifies `melanoma`, `bcc`, or `scc`. The later
T-category model is standalone and was never integrated into the primary
pipeline (`paper/iccit2026/main.tex`, lines 111Ã¢â‚¬â€œ117 and 209Ã¢â‚¬â€œ219).

**FACT.** On the same locked 3,668-image ISIC 2019 internal-test cohort:

| System | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Flat EfficientNet-B0 | 0.742094 | 0.650313 | 0.619222 | 0.752557 |
| Predicted-gate hierarchy | 0.740185 | 0.631199 | 0.605367 | 0.750332 |
| Flat minus hierarchy | +0.001908 | +0.019114 | +0.013855 | +0.002225 |

Sources: `reports/phase07/generated/model_metric_point_estimates.csv`,
`runs/phase05_hierarchical_internal_test/locked_primary_evaluation/hierarchical_metrics.json`,
and `reports/phase06/phase06c_selected_flat_internal_test_result.md`.

**FACT.** The 95% paired, class-stratified bootstrap interval for the
flat-minus-hierarchical macro-F1 difference is `[-0.014255, 0.041963]`.
The accuracy-difference interval is `[-0.011996, 0.015812]`. Exact two-sided
McNemar testing gives `p = 0.820742`, with 354 flat-only correct and 347
hierarchy-only correct images. Sources:
`reports/phase07/generated/bootstrap_confidence_intervals.csv` and
`reports/phase07/generated/paper_table_correctness_agreement.csv`.

**INFERENCE.** The hierarchy did not outperform the flat model on the measured
primary or secondary aggregate metrics. The data also do not establish that
the flat model truly outperforms the hierarchy, because the paired intervals
include zero and only one split/seed was evaluated. Ã¢â‚¬Å“No statistically detected
differenceÃ¢â‚¬Â is supported; equivalence, non-inferiority, and superiority are not.

**FACT.** Oracle routing increases hierarchical macro F1 from 0.605367 to
0.793656, an absolute routing-associated loss of 0.188289. Stage 1 blocks
255/1,270 malignant images (20.079%), incorrectly routes 529/2,398
non-malignant images to Stage 2 (22.060%), and Stage 2 assigns the wrong subtype
to 169/1,015 correctly routed malignant images (16.650%). Source:
`runs/phase05_hierarchical_internal_test/locked_primary_evaluation/hierarchical_metrics.json`.

**INFERENCE.** Routing is the largest measured bottleneck in this implementation.
The oracle result is a diagnostic upper-bound experiment, not a deployable
system and not evidence that a realizable improved gate would reach the oracle
score.

**FACT.** The standalone five-class ISIC-derived T-category experiment used
848 eligible melanoma images and a 594/127/127 train/validation/test split.
The selected inverse-frequency weighted-cross-entropy model achieved test
macro F1 0.275611 and balanced accuracy 0.386039. T2 and T4 recall were zero,
with test supports 7 and 1. Source:
`reports/phase09/isic_stage03_fasttrack_result.md` and
`experiments/evaluations/stage03_isic_derived_wce_internal_test_seed42__best_epoch12/internal_test_metrics.json`.

**INFERENCE.** Stage 3 is feasibility evidence showing severe label-scarcity
failure, not evidence of clinically useful melanoma staging.

### 2.2 Publication verdict

**INFERENCE Ã¢â‚¬â€ reviewer recommendation:** Weak reject in its present form.

The strongest publishable contribution is a careful negative/diagnostic study:
a leakage-aware, same-split comparison showing no established aggregate
advantage for a conditional hierarchy, combined with an oracle-routing
decomposition that identifies error propagation. This is scientifically more
defensible than claiming a superior hierarchical skin-cancer classifier.

The principal publication risks are:

1. one seed and one internal dataset;
2. no completed external evaluation;
3. no calibration, ROC-AUC, PR-AUC, explainability, skin-tone fairness, or
   clinical workflow analysis;
4. only 94 SCC test images;
5. absent patient identifiers, so patient-independent splitting is not proven;
6. conditional hierarchy uses two separately trained backbones and has roughly
   twice the installed parameters/checkpoint storage of the flat model;
7. no FLOPs, peak memory, energy, or rigorous matched latency benchmark;
8. the Ã¢â‚¬Å“lightweightÃ¢â‚¬Â adjective is therefore only partly supported;
9. the manuscript is untracked and repository governance documents are stale.

## 3. Complete repository map

### 3.1 Top-level architecture

| Path | Role | Audit finding |
|---|---|---|
| `README.md` | Entry documentation | Stale: says Phase 02 at line 9 and points to a different source-of-truth directory at line 15. |
| `configs/` | Dataset, experiment, evaluation, and analysis protocols | 17 files; contains frozen protocols and stale project status. |
| `data/` | Checksums and reproducible manifests | Raw/interim/processed/external folders contain only `.gitkeep`; datasets are intentionally not committed. |
| `docs/` | Charter, scope, RQs, governance, risk, decision log | Strong protocol history, but phase labels in early documents are historical/stale. |
| `experiments/` | Registry, Stage-3 runs/evaluations | Includes registry plus locked Phase-9 Stage-3 artifacts. |
| `models/` | Canonical checkpoint placeholder | Only `models/checkpoints/.gitkeep`; actual checkpoints are under `runs/` and `experiments/runs/`. |
| `notebooks/` | Notebook placeholder | No notebooks; only `.gitkeep`. |
| `paper/` | ICCIT manuscript, figures, bibliography, traceability | Entire directory is untracked in current Git status. |
| `reports/` | Dataset audits and Phase 3Ã¢â‚¬â€œ10 evidence | 115 files; main scientific evidence layer. |
| `runs/` | Training/evaluation outputs and backups | 184 files, including 14 `.pt` checkpoints and compressed backups. |
| `scripts/` | CLI orchestration, acquisition, training, evaluation, audits | 35 files including cached bytecode. |
| `src/` | Reusable data/model/training/evaluation/analysis implementation | 65 files including bytecode; no XAI implementation. |
| `tests/` | Unit/integration tests | 60 files including bytecode; broad protocol and metric coverage. |
| `tmp/` | Embedded Python 3.11 runtime and archive | Runtime/temporary material, not research evidence. |
| `.venv/` | Local dependency environment | 34,134 files; dependency implementation, not authored project logic. |
| `backups/` | Phase-9 locked archive | Large `.tgz` and SHA-256 sidecar. |

### 3.2 Source architecture

`src/data/` implements:

- ISIC hierarchical and flat label mapping:
  `src/data/isic2019_dataset.py`, especially `map_flat_diagnosis` at line 72
  and `ISIC2019HierarchicalDataset` at line 89.
- deterministic evaluation and moderate training transforms:
  `src/data/transforms.py`, `build_train_transform` at line 46 and
  `build_eval_transform` at line 78.
- reproducible data loaders: `src/data/dataloaders.py`.
- hierarchy inference loading:
  `src/data/hierarchical_inference_dataset.py` and
  `src/data/hierarchical_dataloader.py`.
- Stage-3 dataset handling: `src/data/emb_stage03.py`.
- class statistics and flat-label audits.

`src/models/efficientnet_baseline.py` exposes one model factory,
`build_efficientnet_b0` (line 13). No alternative backbone, shared encoder,
multi-head network, ensemble, calibration layer, or explainability model is
implemented.

`src/training/` implements:

- a generic epoch engine with AMP support (`engine.py`);
- experiment configuration, optimizer/scheduler/checkpoint handling, early
  stopping, environment capture, and run outputs (`baseline_experiment.py`);
- class-balanced focal loss (`losses.py`, line 12);
- effective-number and inverse-frequency weights
  (`baseline_experiment.py`, lines 329 and 361).

`src/evaluation/` implements:

- standard classification metrics;
- frozen internal-test evaluation;
- hierarchy protocol loading and checkpoint hash verification;
- conditional prediction collection;
- oracle and predicted gating;
- routing and error-propagation decomposition.

`src/analysis/` implements:

- prediction-pair integrity checks;
- fixed-class confusion-matrix metrics;
- class-stratified paired bootstrap;
- exact McNemar testing;
- evidence review and claim locks;
- figure generation;
- static checkpoint/parameter and stored timing audits.

`src/explainability/` contains only `.gitkeep`.

### 3.3 Dataset artifacts

The ISIC pipeline is represented by:

- acquisition script:
  `scripts/download_isic2019_training_data.ps1`;
- source SHA-256 inventory:
  `data/checksums/isic2019/isic2019_source_file_sha256.csv`;
- 25,331-row dataset manifest:
  `data/manifests/isic2019_dataset_manifest.csv`;
- connected-component grouping:
  `data/manifests/isic2019_split_groups_seed42.csv`;
- frozen split:
  `data/manifests/isic2019_train_val_test_split_seed42.csv`;
- dataset, image, duplicate, grouping, split, and class audits under
  `reports/dataset_audits/`.

Stage-3 data are represented by:

- `data/manifests/emb_stage03_dermoscopic_split_seed42.csv`;
- its audit JSON;
- acquisition/audit/split scripts;
- Phase-9 protocol and result reports.

HIBA is protocol-only:

- registry entry and frozen evaluation YAML;
- label-mapping YAML;
- acquisition, inventory, and audit scripts;
- Phase-10 protocols.

No HIBA dataset manifest, approved audit, predictions, or metrics are present.

### 3.4 Reports, figures, tables, and paper resources

Phase 3Ã¢â‚¬â€œ6 reports document baseline, imbalance-aware Stage 2, hierarchical
evaluation, and flat comparison. Phase 7 is the most complete evidence package:
paired predictions, bootstrap replicates, confidence intervals, McNemar results,
routing decomposition, figures, efficiency inventory, claim locks, artifact
hashes, and paper-ready text. Phase 8 audits missing scope. Phase 9 documents
standalone Stage 3. Phase 10 freezes an external-audit protocol but contains no
external result.

Three Phase-7 figures exist in PDF/PNG/SVG:

1. architecture;
2. flat-versus-hierarchy confusion matrices;
3. per-class F1.

The manuscript copies only Figures 1 and 2. Figure 3 is omitted.

`paper/iccit2026/` contains `main.tex`, compiled output, bibliography, claims
traceability, README, and checklist. It is not tracked by Git.

### 3.5 Checkpoints, logs, and backups

Repository inventory found 14 `.pt` checkpoint files. The primary selected
artifacts are:

- Stage 1:
  `runs/phase03_full/.../best_checkpoint.pt`,
  SHA-256 `95e02c...ba3b`;
- Stage 2 class-balanced focal:
  `runs/phase04_cb_focal_full/.../best_checkpoint.pt`,
  SHA-256 `10986d...97fd`;
- flat clean cross-entropy:
  `runs/phase06_full/.../best_checkpoint.pt`,
  SHA-256 `f3d8b8...dc7a`;
- Stage-3 CE and WCE checkpoints under `experiments/runs/`.

Training histories, resolved configs, environment JSON, validation-per-epoch
metrics, and summaries exist for the principal training runs. Phase-6 and
Phase-9 backups are large local archives with SHA-256 files.

### 3.6 Temporary, unused, legacy, missing, and inconsistent artifacts

**FACT Ã¢â‚¬â€ temporary/runtime:**

- `.pytest_cache/`, `__pycache__/`, and `.pyc` files are generated caches.
- `tmp/python311-runtime/` and `tmp/python-3.11.9-embed-amd64.zip` are an
  embedded Python runtime, not a research result.
- `.venv/` is a local environment.

**FACT Ã¢â‚¬â€ placeholders/unused:**

- `notebooks/` is empty.
- `src/explainability/` is empty.
- canonical `models/checkpoints/`, `reports/figures/`, and `reports/tables/`
  contain only `.gitkeep`, while real artifacts are stored elsewhere.
- `data/raw`, `data/interim`, `data/processed`, and `data/external` have no
  committed content.

**FACT Ã¢â‚¬â€ legacy/superseded:**

- generic `configs/experiments/stage01_*` and `stage02_*` remain marked
  `prepared_not_trained`; Phase-specific configs were actually used.
- the registry marks generic EMB Phase 10 Stage-3 as superseded by the
  ISIC-derived Phase-9 fast track.
- Phase-6 contains a preserved externally terminated attempt and a completed
  restart with the same best validation result.
- README and `configs/project.yaml` still say Phase 02.

**FACT Ã¢â‚¬â€ missing/canonical-path issues:**

- `runs/phase06c/.../internal_test_metrics.json` and predictions are referenced
  in registry/protocol evidence but are not currently extracted at the canonical
  path; Phase-7 says the canonical member is preserved in a verified archive.
- the Phase-8 artifact audit records some Phase-5/6 locked artifacts as
  archive-or-restore required.
- no external-evaluation output, XAI output, calibration output, FLOPs,
  peak-memory, energy, or training-time comparison table exists.
- no patient identifiers exist in ISIC metadata.

**INFERENCE.** Artifact preservation is better than average for a student
project, but reproducibility currently depends partly on local untracked
checkpoints/archives and stale path references. A clean clone cannot reproduce
the paper evidence without separately restoring those artifacts and data.

## 4. Research definition

### 4.1 Objective, questions, and hypotheses

**FACT.** The locked objective is to design and evaluate a lightweight
conditional hierarchical dermoscopic classifier and compare it with a flat
classifier under equivalent conditions (`docs/01_scope_lock.md`, lines 15Ã¢â‚¬â€œ30).

**FACT.** RQ1 asks whether the hierarchy improves clinically relevant
performance; its hypothesis predicts improvements in macro F1, balanced
accuracy, malignant recall, minority recall, and failure interpretability
(`docs/02_research_questions.md`, lines 11Ã¢â‚¬â€œ38).

**FACT.** RQ2 asks how much performance is lost through error propagation.
The required oracle-versus-predicted gating is explicitly designed to answer it
(`docs/01_scope_lock.md`, lines 81Ã¢â‚¬â€œ93).

The remaining chartered questions concern imbalance-aware learning, external
generalisation, calibration/confidence, explainability, and efficiency.

### 4.2 Problem statement

**FACT.** The repository frames flat multiclass classification as potentially
hiding clinically important minority-class and routing failure patterns
(`docs/00_project_charter.md`, lines 7Ã¢â‚¬â€œ15).

**INFERENCE.** The scientifically defensible problem is not Ã¢â‚¬Å“hierarchical
classification is better,Ã¢â‚¬Â but Ã¢â‚¬Å“flat and hierarchical organizations expose
different failure structures, and the cost of conditional routing should be
measured.Ã¢â‚¬Â

### 4.3 Motivation

**Clinical motivation Ã¢â‚¬â€ FACT.** Melanoma, BCC, and SCC are separated after a
malignancy gate, with emphasis on malignant and minority-class recall. The
project explicitly excludes clinical deployment or replacement of
dermatologists (`docs/00_project_charter.md`, lines 66Ã¢â‚¬â€œ75).

**Technical motivation Ã¢â‚¬â€ FACT.** Conditional execution can avoid Stage 2 for
predicted non-malignant samples and allows explicit decomposition of screening
and subtype errors.

**Why hierarchical Ã¢â‚¬â€ INFERENCE.** It aligns the computational decision with a
coarse-to-fine malignancy/subtype structure and makes routing failures
observable. The results show interpretive value, not performance superiority.

**Why EfficientNet-B0 Ã¢â‚¬â€ FACT/INFERENCE.** All compared models use
ImageNet-pretrained EfficientNet-B0. The manuscript cites compound scaling and
compact transfer learning (`main.tex`, lines 60Ã¢â‚¬â€œ63). No repository experiment
compares backbones, so EfficientNet-B0 is a controlled engineering choice, not
an empirically established optimum.

**Why leakage-safe split Ã¢â‚¬â€ FACT.** The dataset contains repeated lesion IDs and
exact duplicate images. The grouping audit found 50 duplicate-hash groups,
10 cross-lesion duplicate groups, and 5,057 repeated connected components.
Shared lesion ID or exact SHA-256 relations were transitively grouped before
splitting. This prevents known lesion/hash leakage.

**Why ISIC 2019 Ã¢â‚¬â€ FACT/INFERENCE.** It supplies 25,331 public dermoscopic
training images and labels supporting the required classes and reproducible
development. It does not supply patient IDs and therefore cannot prove
patient-independent evaluation.

### 4.4 Claimed and actual novelty

**FACT.** The repository itself cites prior hierarchical lesion/medical-image
classification (`paper/iccit2026/main.tex`, lines 67Ã¢â‚¬â€œ76). Therefore hierarchy,
EfficientNet, focal loss, effective-number weighting, and oracle gating should
not individually be claimed as novel inventions.

**INFERENCE Ã¢â‚¬â€ genuinely defensible novelty:**

- the particular locked, same-split comparison of a flat EfficientNet-B0 and
  conditional two-stage EfficientNet-B0 system on the selected ISIC classes;
- paired image-level statistical analysis of stored predictions;
- explicit empirical routing decomposition for this system;
- integration of the negative aggregate result with a standalone rare-label
  T-category feasibility result.

**INFERENCE Ã¢â‚¬â€ incremental elements:**

- applying established architectures and loss functions;
- using a clinically intuitive malignancy-to-subtype taxonomy;
- generating standard confusion matrices and bootstrap intervals.

**Do not claim:**

- first hierarchical skin-cancer classifier;
- novel EfficientNet architecture;
- superior classification;
- clinical utility, diagnostic safety, or workflow benefit;
- patient-independent or fully leakage-free evaluation;
- external generalisation;
- a complete three-stage hierarchy;
- lightweight superiority in latency, memory, FLOPs, or storage;
- robust SCC or rare T-category performance.

**Recommended contribution statement:**

> We provide a controlled, leakage-aware comparison of flat and conditional
> two-stage EfficientNet-B0 classifiers on a fixed ISIC 2019 cohort, pair their
> image-level predictions, and quantify the performance ceiling and loss
> associated with routing. The hierarchy did not show a statistically
> distinguishable aggregate advantage, while oracle analysis identified the
> malignancy gate as the dominant measured bottleneck. A separately locked
> T-category feasibility experiment documents the limits imposed by severe
> rare-class scarcity.

## 5. Dataset and leakage audit

### 5.1 ISIC 2019

**FACT.** Original manifest rows: 25,331. Exact unique hashes: 25,281.
One four-image connected component spanning MEL and NV was excluded.
The all-diagnosis split contains 17,731/3,798/3,798 rows, while the primary
four-class experiment contains 17,124/3,668/3,668 rows.

Primary four-class distribution:

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Non-malignant | 11,193 | 2,398 | 2,398 |
| Melanoma | 3,164 | 678 | 678 |
| BCC | 2,327 | 498 | 498 |
| SCC | 440 | 94 | 94 |

Source:
`reports/dataset_audits/isic2019_phase02_class_statistics_seed42.csv`.

**FACT.** Leakage checks report zero overlap in split-group IDs, lesion IDs,
and exact hashes. Source:
`reports/dataset_audits/isic2019_split_audit_seed42.json`,
`leakage_validation`.

**LIMITATION/INFERENCE.** Ã¢â‚¬Å“Leakage-awareÃ¢â‚¬Â or Ã¢â‚¬Å“known-relation-disjointÃ¢â‚¬Â is
accurate. Ã¢â‚¬Å“Leakage-freeÃ¢â‚¬Â is too strong because 2,084 rows lack lesion IDs,
patient IDs are absent, and near-duplicate visual similarity was not reported
as controlled.

### 5.2 Label construction

The flat label order is `non_malignant`, `melanoma`, `bcc`, `scc`.
The hierarchy maps those labels to:

- Stage 1: non-malignant versus malignant;
- Stage 2: melanoma/BCC/SCC among malignant samples.

The class mapping is encoded in the experiment/evaluation YAML and dataset
implementation. AK/BCC/SCC policy decisions should be described exactly from
`scripts/build_isic2019_dataset_manifest.py` rather than with generic
Ã¢â‚¬Å“skin cancerÃ¢â‚¬Â wording.

### 5.3 Stage-3 subset

**FACT.** The EMB repository was not licensed for image use. EMB identifiers
were used only as candidate ISIC IDs; official ISIC metadata and images supplied
provenance and licensing. The final dataset is therefore an Ã¢â‚¬Å“ISIC-derived
melanoma T-category subset,Ã¢â‚¬Â not an EMB image experiment
(`configs/dataset_registry.yaml`, `emb` and `isic_stage03`).

**FACT.** Of 856 candidates, 848 were eligible. Train counts were
Tis 355, T1 184, T2 33, T3 10, T4 12. This is extreme imbalance.

**INFERENCE.** Calling these labels clinical stage would be misleading.
They are broad metadata-derived T categories, and the model is a standalone
image-label feasibility baseline.

### 5.4 External data

**FACT.** HIBA remains `candidate_pending_official_acquisition_audit`,
disabled, and prohibited from training/model selection. No external inference
was run. MRA-MIDAS is not acquired.

**INFERENCE.** Any generalisation claim beyond the locked ISIC cohort is
unsupported.

## 6. Methodology and pipeline

### 6.1 Preprocessing and augmentation

All main comparisons use ImageNet normalization. Deterministic validation/test
preprocessing resizes the shorter side to 256 then center-crops to 224Ãƒâ€”224.
Training uses a moderate fixed policy: random resized crop, horizontal and
vertical flips, Ã‚Â±15Ã‚Â° rotation, and small color jitter. Exact bounds are in the
Phase-3/4/6 YAML files and `src/data/transforms.py`.

### 6.2 Model and optimization

Main models:

- ImageNet-initialized EfficientNet-B0;
- classifier dropout 0.2;
- 2, 3, 4, or 5 output units depending on task.

Full Phase-specific runs use:

- batch size 64;
- maximum 30 epochs;
- AdamW;
- learning rate 0.0003;
- weight decay 0.0001;
- cosine annealing to 0.000001;
- AMP;
- early stopping patience 7;
- checkpoint selection by validation macro F1;
- seed 42.

Stage 1 and flat use cross-entropy. Stage 2 candidates use cross-entropy,
inverse-frequency weighted cross-entropy, or class-balanced focal loss.
The selected Stage 2 uses effective-number weights with `beta=0.9999` and focal
`gamma=2`. Stage 3 compares ordinary CE and inverse-frequency WCE.

### 6.3 Conditional inference

Stage 1 runs on every image. Under predicted gating, argmax malignant invokes
Stage 2 and argmax non-malignant terminates. The Phase-5 evaluator stored Stage-2
outputs for the union of true and predicted malignant samples so oracle and
predicted results could be computed without later re-inference. This explains
why stored Stage-2 execution count (1,799) is not identical to the operational
predicted-gate invocation count (1,544). The distinction is documented in
`reports/phase07/phase07_gate04_independent_evidence_review.md`.

### 6.4 Statistical protocol

Predictions were paired by image ID. The protocol used 10,000
ground-truth-class-stratified paired bootstrap replicates, seed 42, percentile
95% intervals with NumPy linear quantiles. Exact two-sided McNemar tested
paired correctness. No class-wise hypothesis tests or multiplicity correction
were performed.

**INFERENCE.** Stratification stabilizes class representation in bootstrap
replicates, but the resampling unit is image, not lesion/patient component.
Because images within a lesion component can be correlated even though
components are split-disjoint, image-level intervals may be narrower than a
group-aware analysis. The protocol explicitly says not to infer missing group
IDs, so this cannot be repaired without additional manifest linkage/analysis.

## 7. Complete experiment audit

### 7.1 Phase 03 Ã¢â‚¬â€ clean baselines

**Purpose.** Establish Stage-1 binary and Stage-2 malignant-subtype baselines.

**Input.** Frozen seed-42 ISIC split.

**Protocol.** EfficientNet-B0, CE, common augmentation, AdamW/cosine, batch 64,
30-epoch maximum, patience 7, validation macro-F1 selection.

| Task/split | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Stage 1 validation | 0.817884 | 0.828864 | 0.808693 | 0.821588 |
| Stage 1 internal test | 0.787077 | 0.789932 | 0.774783 | 0.790965 |
| Stage 2 validation | 0.861417 | 0.749586 | 0.763902 | 0.858495 |
| Stage 2 internal test | 0.826772 | 0.696166 | 0.712918 | 0.821983 |

Stage-2 test recall: melanoma 0.890855, BCC 0.825301, SCC 0.372340.

**Outputs.** Checkpoints, history, resolved configs, environment, per-epoch
validation metrics, test predictions, confusion matrices, per-class metrics.

**Failure/lesson.** Validation-to-test degradation is material, especially SCC.
The clean Stage-2 baseline is not the selected hierarchy subtype model.

### 7.2 Phase 04 Ã¢â‚¬â€ Stage-2 imbalance-aware learning

**Purpose.** Improve minority SCC performance without changing backbone/split.

Candidates:

| Candidate, validation | Accuracy | Balanced accuracy | Macro F1 | SCC recall | SCC F1 |
|---|---:|---:|---:|---:|---:|
| Clean CE | 0.861417 | 0.749586 | 0.763902 | 0.478723 | 0.535714 |
| Inverse-frequency WCE | 0.864567 | 0.745833 | 0.764999 | 0.468085 | 0.536585 |
| Class-balanced focal | 0.851969 | 0.776287 | 0.776307 | 0.617021 | 0.604167 |

**FACT.** Class-balanced focal was selected by validation macro F1.
Its locked test results were accuracy 0.833858, balanced accuracy 0.722716,
macro F1 0.724875, weighted F1 0.832915, and SCC F1 0.459893.

**INFERENCE.** The selected loss improved the validation macro F1 and validation
SCC metrics but its test SCC F1 was only modestly above clean CE (0.4599 versus
0.4459). This is useful but not robust evidence of a general imbalance solution.

### 7.3 Phase 05 Ã¢â‚¬â€ conditional hierarchy

**Purpose.** Combine frozen Stage 1 and selected Stage 2; quantify standalone,
oracle, predicted-gate, end-to-end, and routing behavior.

**Checkpoint provenance.** Stage-1 SHA-256 `95e02c...ba3b`; Stage-2 SHA-256
`10986d...97fd`.

| Mode | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Stage 1 standalone | 0.786260 | 0.789306 | 0.774009 | 0.790190 |
| Oracle Stage 2 subset | 0.833858 | 0.722716 | 0.724875 | 0.832915 |
| Oracle four-class | 0.942475 | 0.792037 | 0.793656 | 0.942149 |
| Predicted-gate four-class | 0.740185 | 0.631199 | 0.605367 | 0.750332 |

Predicted-gate confusion matrix:

| True \ predicted | Non-malignant | Melanoma | BCC | SCC |
|---|---:|---:|---:|---:|
| Non-malignant | 1869 | 385 | 101 | 43 |
| Melanoma | 177 | 463 | 18 | 20 |
| BCC | 66 | 60 | 349 | 23 |
| SCC | 12 | 15 | 33 | 34 |

**Failure.** An initial evaluation attempt failed before metrics because of an
AMP dtype mismatch; it was preserved and recovery reportedly changed neither
model nor protocol (`experiments/experiment_registry.csv`, Phase-5 notes).

**Lesson.** Stage-1 false negatives are irreversible under conditional routing.
Routing loss dominates measured end-to-end degradation.

### 7.4 Phase 06A / Phase 06 Ã¢â‚¬â€ clean flat four-class model

**Purpose.** Establish a fair same-split flat comparator.

**Protocol.** Four-class EfficientNet-B0, CE, same augmentation/optimizer/
scheduler/selection rule.

**Training history.** Best validation epoch 2; stopped after 9/30 epochs.
Best validation macro F1 0.653572, balanced accuracy 0.679699, accuracy 0.772901,
weighted F1 0.780655.

**Failure case.** The first full attempt was externally terminated after epoch
6. It preserved the same best epoch-2 checkpoint metrics. A completed restart
was retained. This is an infrastructure interruption, not a model failure.

### 7.5 Phase 06B Ã¢â‚¬â€ flat class-balanced focal candidate

**Purpose.** Ensure flat-model loss selection was not unfairly restricted to CE.

**FACT.** Validation macro F1 0.649007, balanced accuracy 0.679346, accuracy
0.757088, weighted F1 0.767136, SCC F1 0.430233; best epoch 11, early stopped
at epoch 18. CE exceeded it by 0.004565 macro F1, so focal was rejected before
test access.

**Lesson.** Imbalance-aware loss did not improve the selected flat validation
objective, although it increased SCC F1 relative to flat CE validation
(0.4302 versus 0.3871). This illustrates the trade-off between primary macro F1
and a single rare class.

### 7.6 Phase 06C Ã¢â‚¬â€ one-time flat internal test

**Purpose.** Evaluate only the validation-selected flat CE checkpoint once.

**FACT.** Accuracy 0.742094, balanced accuracy 0.650313, macro F1 0.619222,
weighted F1 0.752557, mean loss 0.623267.

Flat confusion matrix reconstructed from stored Phase-7 evidence:

| True \ predicted | Non-malignant | Melanoma | BCC | SCC |
|---|---:|---:|---:|---:|
| Non-malignant | 1792 | 408 | 170 | 28 |
| Melanoma | 135 | 511 | 30 | 2 |
| BCC | 27 | 68 | 389 | 14 |
| SCC | 9 | 12 | 43 | 30 |

Source: `reports/phase07/generated/confusion_matrix_flat.csv`. The canonical
Phase-6C metrics file is not currently extracted, but the generated Phase-7
matrix is part of the locked paired-analysis evidence.

### 7.7 Phase 07 Ã¢â‚¬â€ paired evidence closure

**Purpose.** Pair predictions, freeze statistical protocol, calculate
uncertainty and disagreement, audit routing/efficiency, and generate paper
figures/tables.

**FACT.** Pairing passed for 3,668 images. No significance claim was supported
for aggregate flat-versus-hierarchy differences. Per-class point differences
favor:

- hierarchy for non-malignant F1 by 0.004796 and BCC F1 by 0.010203;
- flat for melanoma F1 by 0.031033 and SCC F1 by 0.039386.

Only the exploratory melanoma F1 bootstrap difference interval excludes zero
in the stored unadjusted table. No predeclared class-wise tests or multiplicity
adjustment exist; it must not be promoted as a confirmatory claim.

### 7.8 Phase 08 Ã¢â‚¬â€ scope/evidence audit

**Purpose.** Governance-only audit, no model/dataset execution.

At the time of that report it classified 12 objectives as 2 complete, 5 partial,
4 missing, and 1 blocked. Some Phase-8 statements became historically stale
after Phase 9 completed Stage-3 feasibility. Its still-valid missing areas are
shared/partial-label learning, external evaluation, XAI, calibration, and full
efficiency profiling.

### 7.9 Phase 09 Ã¢â‚¬â€ standalone Stage-3 feasibility

**Purpose.** Audit an officially sourced ISIC-derived melanoma T-category subset
and compare two locked EfficientNet-B0 loss candidates.

| Test model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| CE | 0.606299 | 0.202403 | 0.162838 | 0.480091 |
| Inverse-frequency WCE | 0.543307 | 0.386039 | 0.275611 | 0.493247 |

WCE per-class recall:

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Tis | 77 | 0.645833 | 0.805195 | 0.716763 |
| T1 | 40 | 0.227273 | 0.125000 | 0.161290 |
| T2 | 7 | 0 | 0 | 0 |
| T3 | 2 | 0.333333 | 1.000000 | 0.500000 |
| T4 | 1 | 0 | 0 | 0 |

**INFERENCE.** The T3 recall of 1.0 is based on two samples and is not stable.
Weighted CE trades overall accuracy for macro balance but still fails T2/T4.

### 7.10 Phase 10 and later

Phase 10 contains HIBA acquisition/audit protocols only. Registry entries for
shared three-task models, integrated three-stage evaluation, external
evaluation, XAI, and integrated analyses are blocked, pending, or not started.
They are plans, not completed experiments.

## 8. Unified results and unavailable metrics

### 8.1 Main test comparison

| Experiment | N | Accuracy | Balanced acc. | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Stage-1 CE | 3,668 | 0.787077 | 0.789932 | 0.774783 | 0.790965 |
| Stage-2 CE | 1,270 | 0.826772 | 0.696166 | 0.712918 | 0.821983 |
| Stage-2 CB focal | 1,270 | 0.833858 | 0.722716 | 0.724875 | 0.832915 |
| Hierarchy predicted gate | 3,668 | 0.740185 | 0.631199 | 0.605367 | 0.750332 |
| Hierarchy oracle gate | 3,668 | 0.942475 | 0.792037 | 0.793656 | 0.942149 |
| Flat CE | 3,668 | 0.742094 | 0.650313 | 0.619222 | 0.752557 |
| Stage-3 CE | 127 | 0.606299 | 0.202403 | 0.162838 | 0.480091 |
| Stage-3 WCE | 127 | 0.543307 | 0.386039 | 0.275611 | 0.493247 |

### 8.2 Main per-class test comparison

| Class | Flat precision | Flat recall | Flat F1 | Hier. precision | Hier. recall | Hier. F1 |
|---|---:|---:|---:|---:|---:|---:|
| Non-malignant | 0.912888 | 0.747289 | 0.821830 | 0.879944 | 0.779399 | 0.826625 |
| Melanoma | 0.511512 | 0.753687 | 0.609422 | 0.501625 | 0.682891 | 0.578389 |
| BCC | 0.615506 | 0.781124 | 0.688496 | 0.696607 | 0.700803 | 0.698699 |
| SCC | 0.405405 | 0.319149 | 0.357143 | 0.283333 | 0.361702 | 0.317757 |

### 8.3 Sensitivity and specificity

Per-class recall in the repository is classwise sensitivity. Specificity is not
included in the main generated comparison table and is therefore **unavailable
as a recorded headline metric**. It could be deterministically calculated from
the stored confusion matrices without retraining, but this report does not
invent or silently add metrics.

### 8.4 ROC-AUC and PR-AUC

**Unavailable.** Probability columns exist in prediction artifacts according to
Phase-7 audits, but no locked ROC-AUC/PR-AUC analysis or curves were found.
Computing them is a new analysis requiring a frozen protocol, multiclass
averaging definitions, and careful hierarchy probability semantics.

### 8.5 Parameters, checkpoint size, timing, and compute

| Measure | Flat | Hierarchy | Evidence status |
|---|---:|---:|---|
| Installed parameter elements | 4,012,672 | 8,021,501 | Static checkpoint audit |
| Stored checkpoint bytes | 48,640,921 | 97,234,738 | Static artifact audit |
| Forward passes/image | 1 | 1Ã¢â‚¬â€œ2; stored mean 1.490458 | Architecture/routing count |
| Stored throughput | 119.563 samples/s | 92.051 samples/s | Tesla T4 evaluator logs; limitations |
| Stored evaluator-loop time | 8.364 ms/sample | 10.864 ms/sample | Derived from recorded elapsed time |
| FLOPs/MACs | unavailable | unavailable | Not profiled |
| Peak GPU memory | unavailable | unavailable | Not measured |
| Peak CPU memory | unavailable | unavailable | Not measured |
| Energy/power | unavailable | unavailable | Not measured |
| Training time | unavailable as unified comparison | unavailable | Logs do not provide a paper-ready matched total |

**INFERENCE.** Conditional execution reduces the average number of second-stage
passes compared with always running two models, but this hierarchy is not
lighter than the flat comparator in installed parameters, checkpoint storage,
or recorded throughput. Ã¢â‚¬Å“Lightweight backboneÃ¢â‚¬Â is defensible; Ã¢â‚¬Å“lighter systemÃ¢â‚¬Â
is not.

## 9. Hierarchical analysis

### 9.1 Does hierarchy outperform flat?

No on observed point estimates: flat is +0.001908 accuracy, +0.019114 balanced
accuracy, +0.013855 macro F1, and +0.002225 weighted F1.

No statistically established difference: paired intervals include zero and
McNemar `p=0.820742`.

Per-class trade-offs exist: hierarchy slightly improves non-malignant and BCC F1
point estimates, while flat improves melanoma and SCC F1. These are exploratory.

### 9.2 Error propagation and gate bottleneck

The gate blocks 20.079% of malignant cases. Once blocked, they become
non-malignant final predictions regardless of the Stage-2 model. This creates:

- 177 blocked melanoma cases;
- 66 blocked BCC cases;
- 12 blocked SCC cases,

as visible in the predicted-gate confusion matrixÃ¢â‚¬â„¢s non-malignant column.

Oracle gating removes all non-malignant subtype errors by definition and sends
all true malignant cases to Stage 2, producing a large optimistic ceiling.
The 0.188289 macro-F1 gap is attributable to the routing intervention under
fixed downstream predictions, not solely to malignant false negatives; false
positive routing of non-malignant cases also contributes.

### 9.3 Stage-2 bottleneck

Among correctly routed malignant cases, 16.650% are assigned the wrong subtype.
SCC remains hardest: oracle Stage-2 SCC recall is 0.457447 and F1 0.459893.
Thus a perfect gate would not solve subtype imbalance.

### 9.4 Cancer-specific behavior

- Flat melanoma recall (0.753687) exceeds hierarchical recall (0.682891).
- Hierarchical BCC precision (0.696607) exceeds flat precision (0.615506), but
  its recall is lower.
- Hierarchical SCC recall (0.361702) exceeds flat recall (0.319149), but much
  lower precision yields lower F1.
- Supports, especially SCC=94, make rare-class conclusions unstable.

### 9.5 Conditional efficiency

The hierarchyÃ¢â‚¬â„¢s operational predicted Stage-2 invocation rate is approximately
42.09% (`predicted_malignant_count=1544` over 3668), while the stored union
execution rate used for analysis is 49.05% (1799/3668). Manuscript efficiency
claims must state which rate is meant. The Phase-7 efficiency table uses the
stored 49.05% rate and labels it accordingly.

### 9.6 Explainability and clinical workflow

**FACT.** No XAI implementation or result exists. No clinician evaluation,
decision-curve analysis, triage threshold study, or workflow simulation exists.

**INFERENCE.** The hierarchy is structurally interpretable at the decision-path
level, but that is not the same as image-level explanation or clinical
interpretability. Its malignancy-first organization resembles a triage concept,
yet argmax routing and a 20.08% malignant block rate are not clinically
acceptable evidence.

## 10. Paper-evidence audit

### 10.1 Strong supported claims

- Same-split flat and hierarchy comparison on 3,668 images.
- Exact aggregate point metrics.
- Paired bootstrap intervals and McNemar result.
- No statistically detected overall difference on this split.
- Oracle-versus-predicted routing loss.
- Exact malignant blocking, benign over-routing, and subtype-error counts.
- Leakage control for known lesion IDs and exact hashes.
- Patient independence cannot be guaranteed.
- Stage-3 WCE outperforms CE on validation-selected test macro F1 but remains
  poor for rare classes.
- External/XAI/calibration evidence is absent.

### 10.2 Weak but usable claims

- Ã¢â‚¬Å“LightweightÃ¢â‚¬Â: only when explicitly referring to EfficientNet-B0 backbone or
  conditional pass avoidance, not total system footprint.
- Ã¢â‚¬Å“Clinically meaningful hierarchyÃ¢â‚¬Â: taxonomy is clinically intuitive, but
  workflow utility is untested.
- Ã¢â‚¬Å“Leakage-awareÃ¢â‚¬Â: correct; Ã¢â‚¬Å“leakage-freeÃ¢â‚¬Â is not.
- Ã¢â‚¬Å“Routing is the main measured limitationÃ¢â‚¬Â: supported relative to the
  oracle decomposition, with the word Ã¢â‚¬Å“measured.Ã¢â‚¬Â

### 10.3 Unsupported claims

- better/superior hierarchy;
- external generalisation or robustness;
- calibrated confidence;
- explainability/Grad-CAM benefits;
- fairness or skin-tone performance;
- clinical usefulness, safety, or deployment readiness;
- complete three-stage pipeline;
- patient-level independence;
- computational superiority;
- statistical equivalence/non-inferiority;
- general claims across datasets/seeds/populations.

### 10.4 Missing evidence

- independent external cohort;
- multiple seeds or repeated splits;
- group-aware uncertainty analysis;
- calibration (ECE, Brier, reliability diagrams);
- ROC-AUC and PR-AUC with predeclared averaging;
- operating-point/sensitivity analysis for the malignancy gate;
- threshold ablation versus argmax;
- backbone or capacity-matched ablation;
- hierarchy with shared encoder;
- loss/augmentation ablations beyond limited candidates;
- matched latency with synchronization/warm-up;
- FLOPs, memory, energy;
- XAI protocol and blinded review;
- subgroup/fairness analysis;
- clinical utility analysis.

### 10.5 Manuscript-specific audit

The current manuscript is unusually cautious and its principal numerical claims
match `claims_traceability.md`. Its strongest sentence is the negative result:
Ã¢â‚¬Å“overall performance was not statistically distinguishable on this split.Ã¢â‚¬Â

Issues:

- title says Ã¢â‚¬Å“LightweightÃ¢â‚¬Â without a matched system-level efficiency result;
- abstract may overcompress oracle macro F1 as Ã¢â‚¬Å“routing-related lossÃ¢â‚¬Â without
  clarifying both false-negative and false-positive routing effects;
- only two figures are included; per-class uncertainty figure is available;
- no explicit dataset exclusion mapping for the four-class subset;
- no training run durations/hardware details beyond method summary;
- bibliography/related work is brief for a novelty defense;
- standalone Stage 3 consumes scarce paper space while not being integrated;
- manuscript is untracked and thus not reproducibly versioned.

## 11. ICCIT-style review

### 11.1 Scores (repository-bounded reviewer inference)

| Criterion | Score / 10 | Rationale |
|---|---:|---|
| Novelty | 5 | Routing decomposition/application is useful but methods are established. |
| Technical quality | 7 | Strong protocol/code/evidence discipline; limited model novelty. |
| Methodology | 6 | Fair same-split design; one seed and no patient IDs. |
| Experimental quality | 5 | Good internal audit; missing external/repeated evaluation and key metrics. |
| Writing | 7 | Current manuscript is concise and appropriately cautious. |
| Scientific value | 6 | Honest negative result and error decomposition are informative. |
| Clinical value | 3 | No clinical validation; substantial malignant blocking. |
| Reproducibility | 6 | Rich configs/hashes/tests, but data/checkpoints and paper are not fully tracked/portable. |

**Estimated acceptance probability Ã¢â‚¬â€ INFERENCE:** 25Ã¢â‚¬â€œ40% in present form,
depending on track competitiveness and reviewer appetite for a rigorous negative
result. This is not a statistical prediction.

### 11.2 Major strengths

1. Frozen, same-split comparison and validation-only model selection.
2. Explicit duplicate/lesion connected-component leakage control.
3. One-time locked internal-test protocols.
4. Stored image-level prediction pairing.
5. Predeclared bootstrap/McNemar analysis.
6. Honest oracle routing decomposition.
7. Strong provenance, hashes, reports, and decision log.
8. Manuscript avoids clinical and superiority overclaims.

### 11.3 Major weaknesses

1. Single split, single seed, internal-only main evidence.
2. No external validation despite chartering HIBA as mandatory.
3. Hierarchy does not outperform flat and costs roughly twice installed
   parameters/storage.
4. Patient independence is unknown.
5. Gate blocks one in five malignant lesions.
6. SCC evidence is underpowered/unstable.
7. No threshold/calibration/ROC/PR analysis for a screening gate.
8. No completed XAI or clinical workflow evidence.
9. Standalone Stage 3 is weak and not integrated.
10. Ã¢â‚¬Å“LightweightÃ¢â‚¬Â is inadequately substantiated at system level.

### 11.4 Minor weaknesses

- stale README/project phase;
- historical Phase-8 claims not updated after Phase 9;
- inconsistent artifact locations;
- canonical locked files sometimes archive-only;
- manuscript untracked;
- no notebook/demo (not scientifically required);
- no consolidated model card/data card;
- no formal power analysis;
- no confidence intervals for Stage-3 metrics;
- no explicit license text bundled for all artifacts.

## 12. Prioritized improvement roadmap

| Priority | Improvement | Retraining? | Rewrite? | New analysis? | New experiment? | Figure? | Table? |
|---|---|---:|---:|---:|---:|---:|---:|
| Critical | Complete approved frozen HIBA evaluation | No model training | Yes | Yes | Yes, inference | Yes | Yes |
| Critical | Remove/qualify Ã¢â‚¬Å“lightweightÃ¢â‚¬Â system claim | No | Yes | No | No | No | No |
| Critical | Restore/version all canonical evidence and track paper | No | No | Integrity audit | No | No | Artifact table optional |
| Critical | State one-seed/internal-only limits everywhere | No | Yes | No | No | No | No |
| High | Add multi-seed or repeated-split evaluation | Yes | Yes | Yes | Yes | CI plot | Yes |
| High | Predeclare and evaluate gate thresholds | Usually no retraining | Yes | Yes | Yes, stored logits/inference | ROC/PR/calibration | Yes |
| High | Add ROC-AUC, PR-AUC, specificity, sensitivity, ECE, Brier | No if probabilities complete | Yes | Yes | No/possible inference | Yes | Yes |
| High | Group-aware bootstrap by lesion/split component | No | Yes | Yes | No | Optional | Yes |
| High | Capacity-matched/shared-encoder hierarchy ablation | Yes | Yes | Yes | Yes | Architecture | Yes |
| High | Matched efficiency profiling | No retraining | Yes | Yes | Benchmark | Optional | Yes |
| Medium | Add Grad-CAM with preregistered case selection | No retraining | Yes | Yes | Yes | Yes | Optional |
| Medium | Error audit of 255 blocked malignant cases | No | Yes | Yes | No | Montage optional | Yes |
| Medium | Add subgroup analysis where metadata permits | No | Yes | Yes | No | Optional | Yes |
| Medium | Decide whether Stage 3 belongs in main paper | No | Yes | No | No | No | No |
| Medium | Expand related-work/novelty comparison | No | Yes | Literature work | No | No | Yes |
| Low | Clean stale status documents and artifact paths | No | Documentation | No | No | No | No |
| Low | Add data/model cards and restoration instructions | No | Documentation | No | No | No | No |

## 13. Manuscript support package

### 13.1 Recommended title

**Routing Errors Matter: A Paired Evaluation of Flat and Conditional
Hierarchical Dermoscopic Classifiers**

This removes the weakly supported Ã¢â‚¬Å“lightweightÃ¢â‚¬Â system claim while preserving
the strongest result.

### 13.2 Recommended figures

1. Architecture diagram:
   `reports/phase07/figures/figure01_architecture.pdf`.
2. Row-normalized confusion matrices:
   `reports/phase07/figures/figure02_confusion_matrix_comparison.pdf`.
3. Per-class F1 with intervals:
   `reports/phase07/figures/figure03_per_class_f1.pdf`.

If limited to two figures, keep architecture and confusion matrices; place
per-class F1 in a table or supplement.

### 13.3 Recommended tables

1. Dataset/split/class counts.
2. Main flat-versus-hierarchy metrics with paired difference intervals.
3. Routing decomposition with correct denominators.
4. Per-class precision/recall/F1.
5. Honest efficiency evidence with unavailable cells.
6. Stage-3 feasibility only if retained, clearly separated.

### 13.4 Best discussion points

- Aggregate similarity hides different classwise error distributions.
- Hierarchical organization creates an interpretable failure pathway.
- Oracle routing exposes a high ceiling but is non-deployable.
- Gate improvement is a more direct target than indiscriminate Stage-2 tuning.
- Conditional computation does not automatically mean a lighter total system.
- Leakage control can be strong for known identifiers while patient independence
  remains unresolved.
- Rare-label weighting improves macro balance without solving label scarcity.

### 13.5 Strongest reviewer defense

> The contribution is not a claim that hierarchy is universally superior.
> We froze a fair comparator and test protocol, paired all predictions, and
> report a negative aggregate result without converting non-significance into
> equivalence. The hierarchyÃ¢â‚¬â„¢s value in this study is diagnostic: oracle
> intervention quantifies how much performance is lost at the gate and locates
> a concrete failure mechanism that flat headline metrics do not expose.

### 13.6 Expected reviewer questions and suggested answers

**Q: Why publish a model that does not beat the flat baseline?**
**A:** Because the paired negative result and routing decomposition answer a
different scientific question: whether and where conditional structure fails.
Avoid claiming performance advancement.

**Q: Is the hierarchy actually lightweight?**
**A:** Each backbone is EfficientNet-B0 and Stage 2 is conditional, but the
system has about 8.02M installed parameters versus 4.01M flat and lower stored
throughput. Revise the claim accordingly.

**Q: Why only one seed?**
**A:** The repository cannot defend robustness across seeds. Acknowledge this
as a major limitation or add repeated experiments before submission.

**Q: Why no external dataset?**
**A:** HIBA was deliberately blocked pending provenance/licence/label audit.
This is good governance but leaves generalisation unsupported. Complete it or
scope the paper strictly to internal evaluation.

**Q: Why use accuracy for screening when malignant false negatives matter?**
**A:** Macro F1 and balanced accuracy were primary, and malignant recall is
reported. A clinically oriented paper still needs threshold/sensitivity and
calibration analysis.

**Q: Does oracle-gate loss equal the cost of malignant blocking?**
**A:** Not exclusively. It measures the result of replacing predicted routing
with true routing under fixed Stage-2 predictions, incorporating both malignant
blocking and false-positive non-malignant routing effects.

**Q: Is Stage 3 part of the hierarchy?**
**A:** No. It is a separate feasibility study and must be labeled as such.

**Q: Is the split patient-independent?**
**A:** No guarantee is possible because ISIC 2019 metadata lacks patient IDs.
Known lesion IDs and exact hashes are disjoint.

## 14. Final claim lock

The paper may safely conclude:

1. Flat and predicted-gate hierarchical EfficientNet-B0 systems produced
   similar aggregate performance on one locked ISIC 2019 test cohort.
2. The observed flat point estimates were slightly higher for all four
   aggregate metrics, but paired analysis did not establish an overall
   difference.
3. Oracle routing exposed a 0.188289 macro-F1 gap and a 20.079% malignant
   blocking rate.
4. Routing was the largest measured limitation of the hierarchy.
5. The selected Stage-2 focal loss improved validation macro F1 relative to the
   tested Stage-2 alternatives, but SCC test performance remained weak.
6. The standalone weighted Stage-3 experiment improved macro balance over CE
   but failed rare T2/T4 categories.
7. Results do not establish external generalisation, clinical utility,
   patient-independent performance, calibration, fairness, explainability, or
   system-level computational superiority.

## 15. Audit conclusion

**FACT.** The repository has a strong internal evidence-governance structure:
frozen protocols, deterministic splits, hashes, stored predictions, paired
statistics, claim locks, and explicit negative findings.

**INFERENCE.** Its most credible ICCIT contribution is methodological honesty
and failure decomposition, not a new state-of-the-art classifier. The hierarchy
does not outperform the flat model on the recorded split, and its malignancy
gate is clinically concerning. The standalone T-category experiment reinforces
the limits of rare-label learning rather than completing the hierarchy.

**RECOMMENDATION.** Before submission, prioritize an approved frozen external
evaluation, repeated-seed evidence, threshold/calibration analysis, group-aware
uncertainty, and a defensible efficiency characterization. If these cannot be
completed, narrow the manuscript to a rigorously bounded internal paired study,
remove Ã¢â‚¬Å“lightweightÃ¢â‚¬Â as a comparative system claim, keep Stage 3 secondary, and
present the work explicitly as an error-propagation analysis with a negative
aggregate result.

---

## Appendix A. Primary evidence index

- Governance: `docs/00_project_charter.md`, `docs/01_scope_lock.md`,
  `docs/02_research_questions.md`, `docs/04_reproducibility_protocol.md`,
  `docs/05_risk_register.md`, `docs/06_decision_log.md`.
- Dataset registry: `configs/dataset_registry.yaml`.
- ISIC manifest/split:
  `data/manifests/isic2019_dataset_manifest.csv`,
  `data/manifests/isic2019_split_groups_seed42.csv`,
  `data/manifests/isic2019_train_val_test_split_seed42.csv`.
- Leakage audits:
  `reports/dataset_audits/isic2019_split_group_audit.json`,
  `reports/dataset_audits/isic2019_split_audit_seed42.json`.
- Phase 3:
  `reports/phase03/clean_baseline_internal_evaluation.md`.
- Phase 4:
  `reports/phase04/stage02_imbalance_aware_validation_comparison.md`,
  `reports/phase04/stage02_imbalance_aware_final_internal_evaluation.md`.
- Phase 5:
  `reports/phase05/conditional_hierarchical_internal_evaluation.md`,
  `runs/phase05_hierarchical_internal_test/locked_primary_evaluation/hierarchical_metrics.json`.
- Phase 6:
  `reports/phase06/fair_flat_four_class_protocol.md`,
  `reports/phase06/phase06c_selected_flat_internal_test_result.md`.
- Phase 7:
  `reports/phase07/phase07_final_summary.md`,
  `reports/phase07/generated/statistical_analysis_results.json`,
  `reports/phase07/generated/bootstrap_confidence_intervals.csv`,
  `reports/phase07/generated/model_metric_point_estimates.csv`,
  `reports/phase07/generated/efficiency_comparison_table.csv`.
- Phase 8:
  `reports/phase08/phase08_scope_deviation_and_evidence_audit.md`.
- Phase 9:
  `reports/phase09/isic_stage03_fasttrack_result.md`.
- Phase 10:
  `reports/phase10/hiba_external_dataset_audit_protocol.md`.
- Manuscript:
  `paper/iccit2026/main.tex`,
  `paper/iccit2026/claims_traceability.md`.

## Appendix B. Metrics explicitly unavailable

The following were requested by the audit brief but are not recorded as locked,
repository-resident measurements:

- macro precision and macro recall for the main systems;
- specificity table;
- ROC-AUC;
- PR-AUC;
- calibrated sensitivity/specificity operating points;
- expected calibration error;
- Brier score;
- FLOPs/MACs;
- peak GPU memory;
- peak CPU memory;
- energy/power;
- a unified matched training-time comparison;
- externally validated metrics;
- XAI quality metrics.

They must be reported as unavailable until a frozen analysis or experiment
produces them.
---

## Post-Audit Amendment: Final Phase 11 DenseNet-121 Comparator

**Amendment date:** 2026-07-31
**Locked evidence commit:** `fbd5a37c579a5522878dfe4d97af8efdf5a1f5ee`
**Experiment status:** Completed and locked
**Manuscript source:** Overleaf; local LaTeX sources are intentionally excluded

### Scope

The original scientific audit was completed before execution of the final
protocol-matched DenseNet-121 flat comparator. This amendment supersedes the
original report only where the architecture inventory and final aggregate
comparison changed.

No additional dataset, hyperparameter search, multi-seed experiment,
test-time augmentation, ensemble, threshold tuning, external evaluation, or
internal-test rerun was performed.

### Final Locked Comparison

All three systems were evaluated on the same locked 3,668-image ISIC 2019
internal-test cohort.

| System | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Flat DenseNet-121 | 0.791439 | 0.616828 | 0.635107 | 0.786162 |
| Flat EfficientNet-B0 | 0.742094 | 0.650313 | 0.619222 | 0.752557 |
| Predicted-gate hierarchy | 0.740185 | 0.631199 | 0.605367 | 0.750332 |

DenseNet-121 produced the highest accuracy, macro-F1, and weighted-F1 point
estimates. EfficientNet-B0 retained the highest balanced accuracy.

DenseNet minus EfficientNet-B0:

- accuracy: `+0.049346`;
- balanced accuracy: `-0.033484`;
- macro-F1: `+0.015885`;
- weighted-F1: `+0.033606`.

DenseNet minus predicted-gate hierarchy:

- accuracy: `+0.051254`;
- balanced accuracy: `-0.014371`;
- macro-F1: `+0.029740`;
- weighted-F1: `+0.035831`.

### Class-Level Limitation

DenseNet-121 achieved the following class F1 scores:

- non-malignant: `0.867817`;
- melanoma: `0.602524`;
- BCC: `0.726531`;
- SCC: `0.343558`.

SCC recall remained `0.297872`, with only 28 of 94 SCC images classified
correctly. The higher aggregate scores therefore did not remove the
minority-class sensitivity limitation.

### Statistical Claim Boundary

The existing class-stratified bootstrap confidence intervals and exact
McNemar test apply only to the EfficientNet-B0 flat-versus-hierarchy
comparison.

No new paired confidence interval, McNemar test, multiplicity-adjusted
analysis, or multi-seed analysis was performed for DenseNet-121.

The DenseNet-121 ranking is therefore a descriptive point-estimate comparison.
It must not be presented as statistically significant superiority.

### Hierarchical Interpretation

Oracle routing increased hierarchical macro-F1 from `0.605367` to `0.793656`,
corresponding to a routing-associated loss of `0.188289`.

Stage 1 blocked 255 of 1,270 malignant images, or `20.079%`. Routing therefore
remains the largest measured bottleneck in the conditional hierarchy.

The oracle result is diagnostic and does not imply that a deployable gate
would achieve oracle performance.

### Final Architecture Inventory

The completed repository now contains:

- flat EfficientNet-B0 four-class classification;
- flat DenseNet-121 four-class classification;
- EfficientNet-B0 Stage-1 malignancy classification;
- EfficientNet-B0 Stage-2 malignant-subtype classification;
- predicted and oracle hierarchical evaluation;
- standalone melanoma T-category feasibility evaluation.

The repository does not contain a completed shared-encoder multitask system,
an integrated three-stage pipeline, a completed external-dataset evaluation,
or a clinically validated deployment.

### Evidence

- `reports/phase11/phase11_final_densenet_baseline_result.md`
- `reports/phase11/generated/final_model_comparison.csv`
- `experiments/evaluations/phase11_densenet121_internal_test_seed42__best_epoch04/`
- `experiments/experiment_registry.csv`
- `docs/06_decision_log.md`
- implementation commit `133442a185060d78f23019d6997b12526e6cef3c`
- evidence commit `fbd5a37c579a5522878dfe4d97af8efdf5a1f5ee`

### Manuscript Workflow

The ICCIT manuscript and bibliography are maintained in Overleaf. Local
`main.tex`, `references.bib`, and generated LaTeX build artifacts are
intentionally excluded from this repository.

Repository-resident reports, figures, metrics, predictions, tables, decision
records, and audit evidence remain the source for validating manuscript
claims.

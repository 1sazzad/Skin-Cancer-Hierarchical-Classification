# COM-01B: CoMeSySo 2026 protocol freeze

This phase freezes the study specification only. The companion configuration is
[`comesyso2026_protocol.yaml`](../../../configs/extensions/comesyso2026_probability_fusion/comesyso2026_protocol.yaml).
No inference pipeline or fusion-fitting implementation is introduced. No new
ISIC internal-test or HIBA results are calculated or used to choose methods.
All fitting, inference, training, evaluation, tests, scripts, Modal, and Git
operations are outside this phase. Later execution requires a separate task.

## Scientific identity and provenance

The authoritative repository is
[Skin-Cancer-Hierarchical-Classification](https://github.com/1sazzad/Skin-Cancer-Hierarchical-Classification).
Its `main` branch represents the ICCIT baseline. The separate
`itt2026-consensus-selective-referral` branch contains the ITT extension.
Historical `paul2026_swin` experiments were incorporated into ICCIT; no manuscript
was submitted to PAUL. Preserve `paul2026_*` names only as provenance identifiers.

ICCIT asks: **Does hard hierarchical routing introduce routing-associated
degradation compared with flat classification?** CoMeSySo asks: **Can
probability-preserving post-hoc inference reduce degradation caused by
irreversible hard routing without retraining the underlying hierarchical
representation?** ICCIT supplies the problem diagnosis; CoMeSySo evaluates a new
inference-stage response. Fusion must not be represented as part of ICCIT.

ITT studies cross-backbone consensus, disagreement, majority/weighted ensembles,
and selective referral using seven frozen Flat CNNs. CoMeSySo uses frozen
Shared-Hard Swin-T models. No ITT consensus outputs enter the fusion features.

Read-only inspection used the authoritative repository's six Swin configs under
`configs/extensions/paul2026_swin/`,
`src/evaluation/paul2026_swin_evaluation.py`,
`configs/paper/evaluation/phase06f_hiba_external.yaml`, and
`docs/extensions/paul2026_swin_hiba_evaluation.md` on `main`, plus
`configs/extensions/itt2026_consensus/itt2026_protocol.yaml` on the ITT branch.
These are source/configuration references, not performance-based selection.
Local/remote equivalence and checkpoint bytes were not verified in this phase.

## Frozen model identities

Use exactly seeds **42, 123, 2026**, preserving all underlying Swin-T parameters.
The following selections are already declared by the existing Swin evaluator;
their epochs are provenance, not new checkpoint choices.

| Seed | Shared-Hard run | Epoch | Same-seed Flat run | Epoch |
| --- | --- | --- | --- | --- |
| 42 | `paul2026_shared_hard_swin_t_seed42` | 10 | `paul2026_flat_swin_t_seed42` | 9 |
| 123 | `paul2026_shared_hard_swin_t_seed123` | 1 | `paul2026_flat_swin_t_seed123` | 10 |
| 2026 | `paul2026_shared_hard_swin_t_seed2026` | 26 | `paul2026_flat_swin_t_seed2026` | 17 |

The evaluator resolves each frozen checkpoint as
`results/extensions/paul2026_swin/runs/<run_name>/best_checkpoint.pt`.
Existing locks, summaries, hashes, and seed identities must be verified before
future execution. Missing/mismatched inputs require stopping, not substitution.
No retraining, fine-tuning, checkpoint reselection, architecture modification,
seed replacement, or result-dependent seed dropping is allowed. Preserve the
existing Task-3 head but do not use it as a four-class fusion feature.

## Five systems and probability definitions

The final class order is **0 = non_malignant (NM), 1 = melanoma (MEL),
2 = bcc (BCC), 3 = scc (SCC)**. Task 1 uses `[NM, malignant]`; Task 2 uses
`[MEL, BCC, SCC]`, mapped to final labels `[1, 2, 3]`.

Use uncalibrated softmax probabilities from the frozen heads and preserve
historical deterministic evaluation preprocessing: RGB, antialiased bilinear
short-side resize to 256, center crop 224, and ImageNet normalization. No
augmentation or learned feature scaling is added. Task-2 outputs are needed for
**every sample**, including samples whose Task-1 prediction is NM. On true NM
images these are outputs of the frozen conditional subtype head, not ground-truth
subtype probabilities. Do not mask them using the root decision or true label.
Argmax ties use the first class in the declared order, preserving the existing
Hard procedure. Ground truth may enter prediction only for Oracle.

**A. Flat Swin-T:** use the corresponding same-seed frozen Flat checkpoint and
four-class argmax. This is a contextual comparator, not the new contribution.

**B. Hard Routing:** preserve Shared-Hard inference. Task 1 selects NM or
malignant. An NM decision produces final label 0; a malignant decision produces
the argmax Task-2 subtype, mapped to final labels 1, 2, or 3. This is the ICCIT
hierarchical baseline.

**C. Path-Soft:** compose the probabilities without fitted parameters:

```text
P(NM)  = P_Task1(NM)
P(MEL) = P_Task1(malignant) * P_Task2(MEL | malignant)
P(BCC) = P_Task1(malignant) * P_Task2(BCC | malignant)
P(SCC) = P_Task1(malignant) * P_Task2(SCC | malignant)
prediction = argmax[P(NM), P(MEL), P(BCC), P(SCC)]
```

Each conditional probability is at most 1. Therefore every malignant leaf is
bounded above by `P_Task1(malignant)`. If `P_Task1(NM) > P_Task1(malignant)`,
no malignant leaf can exceed the NM probability. **Path-Soft cannot rescue an
NM-root preference.** Describe it as a parameter-free probabilistic hierarchical
baseline, never as the primary proposed routing-error remedy.

**D. Probability Fusion (primary new method):** independently for each seed,
fit one four-class multinomial logistic-regression classifier on all four classes
of the frozen ISIC validation partition. Its input order is exactly:

```text
x = [P_Task1(malignant), P_Task2(MEL | malignant),
     P_Task2(BCC | malignant), P_Task2(SCC | malignant)]
y = true final class in [0, 1, 2, 3]
```

Freeze L2 penalty, `C=1.0`, `solver=lbfgs`, `fit_intercept=true`,
`max_iter=1000`, and `class_weight=null`. The objective is joint four-class
multinomial softmax, not one-vs-rest. The configuration makes the ordinary
`tol=1e-4` convergence tolerance explicit for reproducibility. There is no
hyperparameter search, threshold search, feature selection, or comparison of
alternative meta-classifiers for selection. Predict by argmax after aligning
classifier outputs to the fixed final class order. Freeze the three resulting
classifiers independently; no seed ensemble or pooled fitting is introduced.
Underlying Swin-T parameters remain frozen. Convergence failure must be reported
and execution stopped without silently changing the solver or specification.
No classifier is fitted during COM-01B.

**E. Oracle Routing:** use the true Task-1 routing label with the same frozen
Task-2 output. True NM gives 0; true malignant gives the predicted subtype.
Oracle remains diagnostic only, non-deployable, and not a proposed practical
system. It references routing-associated headroom; it does not guarantee perfect
subtype classification or constitute a mathematical upper bound on Fusion.

## Cohorts, fitting boundary, and limitation

All model seeds use the same frozen
`data/manifests/isic2019_train_val_test_split_seed42.csv`. Model seed 123 or 2026
does not imply a different data split. Do not regenerate splits or substitute
the ISIC-derived Task-3 cohort for the four-class validation cohort.

ISIC validation is used only to generate the frozen Task-1/Task-2 probabilities
and fit the post-hoc classifier. **This validation partition already contributed
to selection of each underlying Swin-T `best_checkpoint.pt`. Its reuse means
the fusion-fitting cohort is not an independent calibration dataset.** Disclose
this limitation; validation performance is not independent evidence of
generalization. The ISIC internal test remains untouched during method fitting,
and HIBA remains untouched zero-shot external evaluation.

ISIC internal evaluation uses the frozen **3,668-image internal test** and the
fixed four-class order. Internal-test samples cannot influence fusion fitting,
hyperparameters, feature definitions, method selection, thresholds, or checkpoints.

HIBA uses
`data/external/hiba/manifests/hiba_external_dermoscopic_4class_final.csv` unchanged:
**1,232 images, 568 unique patients**, with **696 NM, 196 MEL, 229 BCC, 111 SCC**.
The historical canonical CRLF manifest SHA-256 is
`a2f30f14a249d8acb2bd9f03884e5e41c773ddbee19910d5b739407521e3706a`.
Preserve image identity and patient mapping, including the source `isic_id` and
`patient_id`. Do not add diagnosis filtering. Transfer each seed's exact ISIC
validation coefficients, intercepts, and class mapping unchanged. HIBA fitting,
fine-tuning, calibration, threshold selection, feature selection, checkpoint
selection, and model selection are forbidden.

## Endpoints and metrics

The primary endpoint is **four-class macro-F1**. The primary inferential contrast
on both evaluation cohorts is **Probability Fusion minus Hard Routing**. Its
primary status is fixed regardless of result direction.

Secondary contrasts, with fixed subtraction directions, are:

1. Path-Soft minus Hard.
2. Probability Fusion minus Flat.
3. Oracle minus Probability Fusion.

For every system report accuracy, balanced accuracy, macro-F1, weighted-F1,
per-class precision, recall and F1, and a confusion matrix. Always use labels
`[0, 1, 2, 3]` and `zero_division=0`. Macro-F1 averages all four class F1 values;
balanced accuracy averages all four recalls. Retain absent classes with zero
undefined scores even in bootstrap replicates. Confusion-matrix rows are true
classes and columns predicted classes, in the fixed order.

## Mandatory routing-rescue records and counts

For each dataset, seed, and sample preserve: sample ID, true final class, Task-1
target, Task-1 hard prediction, Task-1 NM and malignant probabilities, all three
Task-2 probabilities, Flat/Hard/Path-Soft/Fusion/Oracle predictions, and
Hard/Path-Soft/Fusion/Oracle correctness. Store dataset and seed explicitly and
require patient ID on HIBA (nullable on ISIC). Correct means **exact final class
matches truth**, not merely correct malignant/NM grouping. Align every system
by the same sample IDs; never silently drop unmatched rows.

For malignant cases incorrectly routed NM (`truth in [1,2,3]` and Task-1
prediction 0), report the total, number corrected by Fusion, number still wrong,
and rescued cases by true subtype. A move from NM to an incorrect malignant
subtype is still wrong and is not a rescue.

For true NM cases incorrectly routed malignant (`truth == 0` and Task-1
prediction 1), report the total, Fusion-corrected count, still-wrong count,
unchanged wrong predictions, and changes to another incorrect class. Hard is
already wrong throughout this subset: a correctness regression cannot occur
within it. Report that made-worse count as zero by definition and separately
report wrong-to-different-wrong changes. Correct-to-wrong regressions remain
mandatory in the full-cohort analysis.

The full-cohort pairwise analysis has five mutually exclusive categories:

1. Hard wrong -> Fusion correct.
2. Hard correct -> Fusion wrong.
3. Hard and Fusion both correct.
4. Hard and Fusion both wrong with the same prediction.
5. Hard and Fusion both wrong with different predictions.

Also explicitly report the requested label **Hard wrong -> Fusion wrong but
prediction changed** as an alias of category 5. These are the same events, so do
not double count them. The five categories must sum to the cohort size for each
seed. These analyses directly assess routing recovery alongside aggregate F1.

## Frozen statistical analysis

Analyze each seed independently on each cohort. For every declared contrast
report its observed macro-F1 difference and paired bootstrap interval in the
declared subtraction direction. Recompute both system metrics on the same
resampled records for each replicate; do not subtract independently generated
intervals. These are percentile intervals, not intervals across pooled seeds.

**Internal ISIC:** paired image-level bootstrap, **10,000 replicates**, statistical
seed **42**, percentile **95% CI** at 2.5th and 97.5th percentiles with linear
quantile interpolation. Sample paired image indices with replacement, preserving
targets and all system predictions together. Apply to Fusion-Hard (primary) and
all three secondary contrasts.

For paired correctness, separately report the **exact two-sided McNemar test**
for each declared comparison and both discordant counts: minuend correct /
subtrahend wrong, and minuend wrong / subtrahend correct. The test uses the exact
binomial null with probability 0.5 conditional on discordance; with no discordant
pairs report p=1. McNemar addresses correctness disagreement, whereas bootstrap
addresses macro-F1 differences. Keep their findings separately labeled.

**HIBA:** patient-cluster paired bootstrap, **5,000 replicates**, statistical seed
**42**, **95% percentile CI**, preserving the existing Swin/HIBA settings. Use
sorted unique patient IDs and the historical NumPy generator. Draw 568 patients
with replacement per replicate; include every image of each selected patient,
including repeated copies when a patient is sampled more than once. Pair all
systems on identical patient draws. Recompute the image-based four-class metric
on each resulting resampled collection; do not average patient F1 scores. Use
linear 2.5th/97.5th percentiles. Apply Fusion-Hard as primary and the same three
secondary contrasts. Ordinary image independence and image-level McNemar are
not the primary HIBA uncertainty analysis. Preserve other frozen Swin/HIBA
settings unless this protocol explicitly supersedes them.

**Three-seed reporting:** report seeds 42, 123, and 2026 separately, then arithmetic
mean and sample standard deviation (`ddof=1`) of scalar metrics and contrast
estimates across those three seeds. Preserve each seed's intervals, tests,
confusion matrix, per-class metrics, and rescue counts. Across-seed spread is
descriptive and does not replace paired inference. Never pool seed-image pairs
as independent observations, omit an unfavorable seed, or choose a representative
seed based on test performance.

## Leakage prohibitions and output boundary

Forbidden are fusion fitting on internal test or HIBA; HIBA calibration; tuning
C from test/HIBA results; changing the solver after results; choosing alternative
meta-classifiers from test/HIBA performance; test/HIBA feature or threshold
selection; dropping/replacing seeds; checkpoint reselection; Swin-T retraining,
fine-tuning, or architecture changes; changing the primary comparison or
Path-Soft/Fusion definitions after results; and overwriting ICCIT or ITT results.
The classifier specification also prohibits searches on validation.

All future CoMeSySo artifacts must stay within:

- `configs/extensions/comesyso2026_probability_fusion/`
- `docs/extensions/comesyso2026_probability_fusion/`
- `results/extensions/comesyso2026_probability_fusion/`
- `scripts/comesyso2026/`

Historical checkpoints and provenance files are read-only inputs. Do not modify
`results/paper/`, `configs/paper/`, ICCIT manuscript artifacts,
`results/extensions/itt2026_consensus/`,
`configs/extensions/itt2026_consensus/`, or ITT manuscript artifacts.
Only this document and the companion YAML are created in COM-01B.

## Inspection findings and verification limits

No conflicting frozen HIBA statistical requirement was found: both the HIBA
config and Swin HIBA protocol support 5,000 patient-cluster replicates, seed 42,
and confidence level 0.95. ITT's 10,000-replicate setting belongs to a separate
study and does not supersede the frozen Swin/HIBA configuration.

The original Swin training configs retain `ready_for_training` and
`runs/extensions/paul2026_swin/` as their output root. The frozen evaluator
instead declares selected epochs and reads checkpoints under
`results/extensions/paul2026_swin/runs/`. This protocol follows the evaluator's
frozen input identities; legacy training metadata is not authorization to train.
The existing Shared-Hard config evaluates Task 2 on malignant validation cases
for checkpoint selection; future Fusion fitting requires Task-2 outputs for all
four-class validation cases. This is a new inference-data requirement, not a
change to the historical training or checkpoint selection protocol.

Repository inspection was through read-only GitHub file access. No local/remote
identity, checkpoint availability, YAML parsing, or automated verification is
claimed. Two terminal reads occurred while initially opening the attached request
and searching for repository instructions, before its no-terminal restriction
was seen; no further terminal command was used. No tests, training, fitting,
inference, evaluation, Modal, package installation, or Git operation was run.

The user may later run these read-only verification commands from the repository
root. They are supplied for manual use and were not executed here. The YAML
command assumes PyYAML is already available; no installation is requested.

```powershell
git status --short
git diff --check
git diff -- configs/extensions/comesyso2026_probability_fusion/comesyso2026_protocol.yaml docs/extensions/comesyso2026_probability_fusion/COM01_PROTOCOL_FREEZE.md
Get-Content -LiteralPath 'configs/extensions/comesyso2026_probability_fusion/comesyso2026_protocol.yaml'
Get-Content -LiteralPath 'docs/extensions/comesyso2026_probability_fusion/COM01_PROTOCOL_FREEZE.md'
python -c "from pathlib import Path; import yaml; p=Path('configs/extensions/comesyso2026_probability_fusion/comesyso2026_protocol.yaml'); d=yaml.safe_load(p.read_text(encoding='utf-8')); assert d['protocol']['phase']=='COM-01B'; assert d['models']['seeds']==[42,123,2026]; assert len(d['systems'])==5; print('YAML parsed; phase, seeds, and system count checked')"
```

`git diff` does not display untracked new files; the two `Get-Content` commands
allow reviewing their contents without staging. YAML parsing is only a syntax
and basic-field check, not scientific or runtime validation. No CoMeSySo runner
exists in this phase, and no execution command for one is prescribed.

# Phase 06 — Gate 06B Results and Evaluation Integration

Date: 2026-08-29

## Gate status

**Gate 06B: PASS**

Gate 06B integrated the frozen Phase 05 scientific evidence into the manuscript Results and Evaluation protocol. No training, inference rerun, checkpoint replacement, threshold tuning, calibration, preprocessing change, post-hoc optimisation, or model reselection was performed.

## Files changed

- `paper/sections/results_discussion.tex`
- `paper/sections/methodology.tex`

## Results integration completed

The obsolete ICCIT-era three-system Results story was removed from the active Results section. The manuscript now uses the Phase 05 canonical comparison:

- Flat four-class macro-F1: `0.619222`
- Shared predicted-gate hierarchy macro-F1: `0.568591`
- Shared oracle-gate hierarchy macro-F1: `0.776976`
- Routing-associated gap: `0.208385`

The final end-to-end table now reports:

| System | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Flat four-class | 0.742094 | 0.650313 | 0.619222 | 0.752557 |
| Shared hierarchy — predicted gate | 0.687568 | 0.639536 | 0.568591 | 0.706832 |
| Shared hierarchy — oracle gate | 0.933751 | 0.779171 | 0.776976 | 0.933991 |

Oracle routing is explicitly labeled diagnostic and non-deployable.

## Paired statistical evidence integrated

The manuscript now reports:

- hierarchy minus flat macro-F1 = `-0.050632`
- 95% CI = `[-0.075993, -0.024827]`
- hierarchy minus flat accuracy = `-0.054526`
- 95% CI = `[-0.068702, -0.040349]`

Paired correctness counts:

- both correct = `2248`
- hierarchy only correct = `274`
- flat only correct = `474`
- both wrong = `672`
- exact two-sided McNemar `p = 2.49118e-13`

The manuscript explicitly states that McNemar supports paired correctness / accuracy discordance and is not a significance test for macro-F1.

## Classwise evidence integrated

The Results section now reports the frozen flat-versus-shared-hierarchy F1 evidence:

- Non-malignant: hierarchy `0.778597`, flat `0.821830`, H-F `-0.043233`, CI `[-0.056035, -0.030510]`
- Melanoma: hierarchy `0.545669`, flat `0.609422`, H-F `-0.063753`, CI `[-0.088293, -0.039765]`
- BCC: hierarchy `0.659189`, flat `0.688496`, H-F `-0.029307`, CI `[-0.056238, -0.002722]`
- SCC: hierarchy `0.290909`, flat `0.357143`, H-F `-0.066234`, CI `[-0.150000, 0.019586]`

The SCC qualifier is preserved: the point estimate favors flat, but the confidence interval crosses zero and support is only `94`; no reliable SCC-specific superiority is claimed.

## Routing decomposition integrated

The manuscript now reports the canonical shared-hierarchy routing counts:

- malignant blocked by Task 1: `170 / 1270 = 13.39%`
- non-malignant incorrectly routed to Task 2: `761 / 2398 = 31.73%`
- subtype errors after correct malignant routing: `215 / 1100 = 19.55%`
- predicted-gate macro-F1: `0.568591`
- oracle-gate macro-F1: `0.776976`
- routing-associated gap: `0.208385`

The interpretation is bounded to routing being a **major** end-to-end bottleneck, not the sole proven cause.

Conditional Task 2 evidence was added:

- shared Task 2 malignant-subset macro-F1 = `0.702634`
- accuracy = `0.808661`

This is clearly distinguished from deployed four-class hierarchical performance.

## Shared-versus-standalone evidence integrated

The manuscript now includes:

| Task | Shared macro-F1 | Standalone macro-F1 | Shared - Standalone |
|---|---:|---:|---:|
| Task 1 | 0.740624 | 0.774009 | -0.033385 |
| Task 2 | 0.702634 | 0.724875 | -0.022241 |
| Task 3 | 0.298710 | 0.275611 | +0.023099 |

Task 1 and Task 2 are described only as consistent with possible negative transfer or insufficient specialization. Causation is not claimed.

Task 3 remains explicitly support-limited because T2/T3/T4 supports are `7/2/1`.

## Validation-to-test consistency integrated

The manuscript now reports directional within-study consistency only:

- validation H-F macro-F1 = `-0.070285`
- internal-test H-F macro-F1 = `-0.050632`
- validation H-F accuracy = `-0.063795`
- internal-test H-F accuracy = `-0.054526`
- validation routing loss = `0.192480`
- internal-test routing loss = `0.208385`

The text explicitly states that test results were not used for further model selection or modification.

## Evaluation protocol revision

`paper/sections/methodology.tex` now identifies the publication-facing systems as:

- flat four-class classifier;
- shared representation with task-specific heads;
- deployed hard routing through Task 1 and Task 2;
- Task 3 as a secondary task rather than part of the four-class endpoint path;
- oracle routing as diagnostic only.

The frozen-test rule is now stated explicitly in the manuscript.

The previous methodology description of the final hierarchy as two independently trained EfficientNet-B0 models and the DenseNet-121 comparator as a principal final system has been removed from the active Methodology section.

## Guardrail audit

Gate 06B preserves the following mandatory publication constraints:

- no SOTA claim;
- no clinical superiority/readiness/safety claim;
- no external-generalization claim;
- no causal negative-transfer claim;
- no reliable SCC-specific superiority claim;
- no stable rare Task 3 class claim;
- oracle performance is not presented as deployable;
- routing is not presented as the sole cause;
- McNemar is not used to claim macro-F1 significance.

## Known remaining manuscript inconsistencies

Gate 06B intentionally did not fully rewrite later/earlier narrative sections. The following still require subsequent gates:

- `paper/sections/abstract.tex` still contains obsolete ICCIT-era numbers and system identities.
- `paper/sections/introduction.tex` still frames the older EfficientNet/DenseNet comparison.
- `paper/sections/related_work.tex` still contains outdated comparator positioning.
- `paper/sections/problem_statement.tex` still defines the old final system formulation.
- `paper/sections/conclusion.tex` still contains obsolete results.
- current paper figures are ICCIT-era figures and have not yet been replaced by the approved Phase 05 publication figures.
- the Results section still contains a Limitations subsection whose final narrative alignment belongs to Gate 06C.

These are not Gate 06B failures because they are assigned to Gates 06C--06E.

## Gate 06B verdict

**PASS.**

The manuscript Results and Evaluation protocol now use the frozen Phase 05 evidence, correct paired statistical interpretation, classwise qualifiers, routing decomposition, oracle diagnostic, shared-versus-standalone task evidence, and frozen-test guardrails.

## Exact next gate

**Gate 06C — Discussion, Limitations, and Contribution Revision**

Gate 06C should center the narrative on the flat deployed advantage, routing bottleneck, retained conditional capability, possible but non-causal negative transfer, SCC uncertainty, Task 3 sparsity, and storage/performance trade-off.
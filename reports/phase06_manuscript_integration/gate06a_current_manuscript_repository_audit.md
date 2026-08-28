# Phase 06 — Gate 06A Current Manuscript and Repository Audit

Date: 2026-08-28

## Gate status

**Gate 06A: PASS — AUDIT COMPLETE / NO MANUSCRIPT BODY EDITS PERFORMED**

This gate audited the manuscript state against the frozen Phase 05 scientific evidence before any large-scale paper revision. No training, inference, threshold tuning, checkpoint reselection, calibration, preprocessing change, or test reuse was performed.

## Repository and manuscript source determination

Repository:

`1sazzad/Skin-Cancer-Hierarchical-Classification`

Authoritative local workspace specified by the project workflow:

`F:\Research\Final Year\Skin-Cancer-Hierarchical-Classification-main`

Phase 06 working branch:

`phase06-manuscript-integration`

Phase 06 branch starting point:

`fff4afef72c210837b1ad1c352e9c64b1994aeeb`

The repository contains a complete LaTeX manuscript under:

`paper/`

The active repository manuscript entry point is:

`paper/main.tex`

It includes:

- `paper/sections/abstract.tex`
- `paper/sections/introduction.tex`
- `paper/sections/related_work.tex`
- `paper/sections/problem_statement.tex`
- `paper/sections/methodology.tex`
- `paper/sections/results_discussion.tex`
- `paper/sections/conclusion.tex`
- `paper/references.bib`

A compiled historical PDF also exists at:

`paper/main.pdf`

The repository additionally contains:

`paper/iccit2026/`

This directory contains ICCIT-era submission support material (`README.md`, claim traceability, checklist, and figures). It should be treated as historical submission material rather than the Phase 06 authoritative scientific evidence source.

The project has previously used Overleaf, but Gate 06A did not have an authenticated Overleaf connector or direct project snapshot with which to prove whether the current Overleaf copy is byte-for-byte synchronized with the repository. Therefore:

- the manuscript is **not Overleaf-only**;
- a complete source exists in Git;
- for Phase 06, the repository `paper/` tree should be treated as the authoritative editable source unless a newer Overleaf-only revision is explicitly supplied and reconciled later.

## Current title and framing audit

Current title:

> Routing Errors Matter: An Empirical Analysis of Conditional Hierarchical Dermoscopic Classification

### Status: REVISE, NOT NECESSARILY REPLACE

The title is directionally compatible with the final routing-bottleneck story, but the current manuscript underneath it describes an older experimental system. The final paper must explicitly align with the Phase 05 framing:

**controlled comparative and failure-analysis study**

The title should not imply that the old independently trained two-stage EfficientNet hierarchy is the final evaluated hierarchy.

## Section-by-section audit

### 1. Abstract

**Status: REPLACE SUBSTANTIALLY**

The abstract is materially outdated.

It currently reports:

- flat DenseNet-121 accuracy `0.7914`;
- flat DenseNet-121 macro-F1 `0.6351`;
- predicted-gate hierarchy macro-F1 `0.6054`;
- backbone-matched flat-vs-hierarchy macro-F1 difference `0.0139` with CI crossing zero;
- McNemar `p = 0.8207`;
- oracle-routing macro-F1 `0.7937`;
- routing-associated loss `0.1883`;
- Stage 1 malignant blocking `20.079%`.

These are not the Phase 05 canonical final comparative results.

The Phase 05 frozen headline evidence is:

- flat four-class macro-F1 `0.619222`;
- shared predicted-gate hierarchy macro-F1 `0.568591`;
- hierarchy-minus-flat macro-F1 `-0.050632`, 95% CI `[-0.075993, -0.024827]`;
- hierarchy-minus-flat accuracy `-0.054526`, 95% CI `[-0.068702, -0.040349]`;
- McNemar hierarchy-only correct `274`, flat-only correct `474`, exact two-sided `p = 2.49118e-13`;
- oracle-gate hierarchy macro-F1 `0.776976`;
- routing loss `0.208385`;
- malignant blocked by Task 1 `170/1270 = 13.39%`;
- non-malignant incorrectly routed to Task 2 `761/2398 = 31.73%`.

The current abstract therefore conflicts with the final evidence in both system identity and statistical conclusion.

### 2. Introduction

**Status: MAJOR REVISION**

Parts of the motivation remain useful:

- class imbalance;
- hard routing and error propagation;
- distinction between endpoint and conditional performance;
- oracle routing as diagnostic only.

However, the current introduction frames the principal comparison as:

- flat EfficientNet-B0;
- conditional two-stage EfficientNet-B0 hierarchy;
- protocol-matched DenseNet-121 comparator.

It explicitly states that the backbone-matched comparison does not isolate task decomposition because Stage 2 uses class-balanced focal loss while the flat model uses cross-entropy. This is an ICCIT-era experimental framing and does not describe the final Phase 05 shared multi-head system.

The contribution list is likewise outdated because it focuses on three old systems rather than:

- flat four-class vs shared predicted-gate hierarchy;
- predicted-vs-oracle routing decomposition;
- shared-vs-standalone Task 1/2/3 comparison;
- storage/performance trade-off;
- frozen paired statistical evaluation.

### 3. Related Work

**Status: MODERATE REVISION**

The cited background on:

- EfficientNet;
- DenseNet;
- focal/class-balanced loss;
- hierarchical classification;
- LightDPH;
- two-step hierarchical skin-lesion systems;
- paired bootstrap;
- McNemar

is broadly relevant.

However, the prose currently positions EfficientNet-B0 as the backbone-matched hierarchy comparison and DenseNet-121 as a protocol-matched alternative backbone. This is no longer the final scientific positioning.

Related Work should instead support the paper's final novelty boundary:

- hard conditional routing and error propagation;
- multi-task/shared representation versus independently specialized tasks;
- diagnostic oracle routing;
- failure analysis rather than leaderboard superiority.

No SOTA claim should be introduced.

### 4. Problem Formulation

**Status: MAJOR REVISION**

The current formulation defines:

- two flat implementations, `f_flat,E` and `f_flat,D`;
- a two-model hierarchy `f1` and `f2`;
- standalone T-category task.

This does not match the final shared hierarchy formulation.

The revised formulation should distinguish:

- flat four-class model;
- shared encoder with task-specific heads;
- deployed hard routing through shared Task 1 and Task 2;
- oracle gate as diagnostic;
- Task 3 as a shared/standalone secondary task with severe support limitations.

### 5. Methodology

**Status: MAJOR REVISION / PARTIAL RETENTION**

Likely retain after evidence verification:

- ISIC 2019 source and class mapping;
- exclusion of AK and conflicting duplicate components;
- leakage-aware split language;
- train/validation/test counts;
- preprocessing protocol where unchanged;
- locked-test policy;
- statistical method description (10,000 ground-truth-class-stratified paired bootstrap replicates, seed 42; exact two-sided McNemar for paired correctness).

Materially outdated content includes:

- architecture figure describing flat EfficientNet-B0 vs two independently trained EfficientNet-B0 hierarchy models;
- DenseNet-121 as a principal comparator;
- selected Stage 2 class-balanced focal setup as the deployed final hierarchy definition;
- old checkpoint epoch identities as if they define the Phase 05 final shared hierarchy;
- T-category description as only a standalone EfficientNet-B0 experiment, without the shared Task 3 comparison now present in Phase 05.

The evaluation subsection also needs explicit wording that McNemar supports paired correctness/accuracy discordance and does not test macro-F1 significance.

### 6. Evaluation Protocol

**Status: REVISE AND STRENGTHEN**

The current manuscript contains the core statistical machinery but does not present the final frozen test-use guardrail with enough prominence.

The final manuscript should state clearly:

- internal test `N = 3668`;
- test consumed once under frozen protocol;
- no test-driven threshold tuning, calibration, architecture selection, checkpoint selection, preprocessing change, retraining, or reselection;
- paired bootstrap is used for metric differences;
- McNemar is interpreted only for correctness/accuracy discordance.

### 7. Results

**Status: REPLACE**

The current primary table contains an obsolete three-system comparison:

- DenseNet-121: macro-F1 `0.6351`;
- EfficientNet-B0: macro-F1 `0.6192`;
- hierarchy: macro-F1 `0.6054`.

The current paper then concludes that DenseNet-121 has the strongest aggregate point estimates and that flat EfficientNet-B0 vs hierarchy is statistically inconclusive.

This directly conflicts with Phase 05.

The canonical Phase 05 primary table is:

| System | N | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Flat four-class | 3668 | 0.742094 | 0.650313 | 0.619222 | 0.752557 |
| Shared hierarchy — predicted gate | 3668 | 0.687568 | 0.639536 | 0.568591 | 0.706832 |
| Shared hierarchy — oracle gate | 3668 | 0.933751 | 0.779171 | 0.776976 | 0.933991 |

The final main comparison must report the paired hierarchy-minus-flat macro-F1 and accuracy confidence intervals and the final McNemar discordance counts.

### 8. Classwise Results

**Status: REPLACE**

The current classwise table mixes DenseNet-121, flat EfficientNet-B0, and the old hierarchy.

The final classwise evidence is the flat-vs-shared-hierarchy comparison:

- non-malignant: hierarchy F1 `0.778597`, flat F1 `0.821830`, delta `-0.043233`, CI `[-0.056035, -0.030510]`;
- melanoma: hierarchy F1 `0.545669`, flat F1 `0.609422`, delta `-0.063753`, CI `[-0.088293, -0.039765]`;
- BCC: hierarchy F1 `0.659189`, flat F1 `0.688496`, delta `-0.029307`, CI `[-0.056238, -0.002722]`;
- SCC: hierarchy F1 `0.290909`, flat F1 `0.357143`, delta `-0.066234`, CI `[-0.150000, 0.019586]`.

The manuscript must preserve the SCC qualifier: the point estimate favors flat, but the interval crosses zero and support is only `94`.

### 9. Routing Analysis

**Status: REPLACE NUMBERS / RETAIN CONCEPT**

The current manuscript reports:

- predicted-gate macro-F1 `0.605367`;
- oracle macro-F1 `0.793656`;
- routing loss `0.188289`;
- malignant blocked `255/1270 = 20.079%`;
- non-malignant sent to Stage 2 `529/2398 = 22.060%`;
- wrong subtype after correct route `169/1015 = 16.650%`.

The Phase 05 canonical shared-hierarchy routing decomposition is:

- predicted-gate macro-F1 `0.568591`;
- oracle-gate macro-F1 `0.776976`;
- routing loss `0.208385`;
- malignant blocked `170/1270 = 13.39%`;
- non-malignant incorrectly routed `761/2398 = 31.73%`;
- wrong subtype after correct malignant route `215/1100 = 19.55%`.

The interpretation must remain bounded: routing is a **major** bottleneck, not proven to be the sole cause of underperformance.

### 10. Shared-vs-Standalone Evidence

**Status: MISSING FROM CURRENT MANUSCRIPT / MUST ADD**

The current paper does not integrate the final Phase 05 task-sharing evidence:

- Task 1 shared macro-F1 `0.740624` vs standalone `0.774009`, delta `-0.033385`;
- Task 2 shared macro-F1 `0.702634` vs standalone `0.724875`, delta `-0.022241`;
- Task 3 shared macro-F1 `0.298710` vs standalone `0.275611`, delta `+0.023099`.

Required interpretation:

- Task 1/2 are consistent with possible negative transfer or insufficient specialization;
- causation is not established;
- Task 3 is unstable because T2/T3/T4 supports are `7/2/1`.

### 11. Discussion

**Status: REWRITE**

The current discussion is anchored to the old EfficientNet-vs-EfficientNet inconclusive result and DenseNet comparisons.

The final discussion must instead center on:

1. flat four-class is the stronger deployed system under the frozen internal evaluation;
2. shared predicted-gate hierarchy underperforms in macro-F1 and accuracy with paired CIs excluding zero;
3. oracle routing reveals substantial latent conditional capability;
4. routing is a major end-to-end bottleneck;
5. Task 1/2 shared-vs-standalone gaps suggest possible negative transfer but are non-causal;
6. SCC uncertainty prevents reliable SCC-specific superiority claims;
7. Task 3 rare categories are too sparse for stable conclusions;
8. shared architecture offers storage consolidation, but isolated forward-pass evidence does not prove full deployment-speed superiority.

### 12. Limitations

**Status: RETAIN CORE, EXPAND/ALIGN**

The existing limitations already correctly mention:

- one internal cohort;
- lack of external validation;
- leakage-aware but not fully patient-independent split;
- SCC support limitations;
- sparse T-category data;
- lack of calibration, fairness, and explainability studies;
- no clinical readiness claim.

They must be expanded/aligned with Phase 05 to include:

- one-time test consumption;
- bootstrap uncertainty does not include retraining/seed/dataset-shift uncertainty;
- McNemar interpretation boundary;
- shared-vs-standalone comparisons are descriptive and non-causal;
- efficiency benchmark limitations.

### 13. Conclusion

**Status: REPLACE**

The current conclusion states that:

- DenseNet-121 has the highest point estimates;
- hierarchy does not improve upon flat EfficientNet-B0;
- the flat-vs-hierarchy paired analysis found no statistically significant difference;
- oracle routing raises macro-F1 from `0.6054` to `0.7937`.

These statements are obsolete for the Phase 05 final system.

The final conclusion must state that:

- flat four-class macro-F1 is `0.619222`;
- shared predicted-gate hierarchy macro-F1 is `0.568591`;
- hierarchy-minus-flat macro-F1 is `-0.050632`, 95% CI excluding zero;
- accuracy evidence and paired correctness point in the same direction;
- oracle routing raises macro-F1 to `0.776976` and reveals routing loss `0.208385`;
- the hierarchy retains conditional capability despite weaker deployed performance;
- no external/clinical generalization claim is justified.

## Figures audit

Current manuscript figure files:

- `paper/figures/figure01_architecture.pdf`
- `paper/figures/figure02_confusion_matrix_comparison.pdf`

These are ICCIT-era figures and are not the final Phase 05 publication figure set.

Phase 05 canonical figures are:

- `reports/phase05_final_scientific_synthesis/figures/fig1_end_to_end_macro_f1.png`
- `reports/phase05_final_scientific_synthesis/figures/fig2_routing_loss.png`
- `reports/phase05_final_scientific_synthesis/figures/fig3_classwise_f1.png`
- `reports/phase05_final_scientific_synthesis/figures/fig4_classwise_delta_ci.png`

Phase 06 should integrate these approved figures or publication-equivalent copies into the paper tree while preserving their canonical values and captions.

The old architecture figure may still be useful only if it is redrawn to match the final shared multi-head architecture; it must not remain as a depiction of the final system in its current form.

## Tables audit

The current manuscript tables are not Phase 05 canonical.

Authoritative publication-ready tables are in:

`reports/phase05_final_scientific_synthesis/gate05b_publication_tables.md`

Required integration priorities:

1. end-to-end four-class performance;
2. paired flat-vs-hierarchy comparison;
3. per-class F1 comparison;
4. routing decomposition;
5. shared-vs-standalone task performance.

## References audit

The current bibliography contains relevant foundational and comparative references for:

- ISIC/HAM10000/BCN20000;
- EfficientNet and DenseNet;
- focal and class-balanced loss;
- hierarchical classification;
- hierarchy-aware and LightDPH skin-lesion methods;
- two-step hierarchical skin-lesion classification;
- bootstrap methods;
- McNemar testing.

No obvious citation-breaking error was found in Gate 06A.

However, citation coverage must be re-audited after the manuscript is reframed, particularly for:

- multi-task/shared-representation learning and negative transfer terminology;
- hard routing/error propagation;
- clinical/external validation limitations if those claims require support;
- any newly introduced efficiency/storage discussion.

No new citation should be added merely to support a SOTA or clinical claim, because such claims are prohibited by the Phase 05 guardrails.

## Old ICCIT-era conflicts that must be removed or quarantined

The following old scientific storyline is no longer acceptable as the primary paper narrative:

- DenseNet-121 as the headline best deployed model;
- the primary hierarchy being two independently trained EfficientNet-B0 stages;
- flat EfficientNet-B0 versus hierarchy macro-F1 difference being approximately `0.0139` and inconclusive;
- McNemar `p = 0.8207` for the headline flat-vs-hierarchy comparison;
- predicted hierarchy macro-F1 approximately `0.6054`;
- oracle routing approximately `0.7937`;
- routing loss approximately `0.1883`;
- malignant block rate `20.079%`;
- non-malignant false-route rate `22.060%`.

These values may remain in historical audit records but must not remain as current manuscript results.

## Scientific framing lock for Phase 06

The manuscript must be positioned as a:

**controlled comparative and failure-analysis study**

Primary message:

> Under a frozen internal evaluation protocol, the flat four-class classifier provided stronger deployable performance than the shared hard-routing hierarchy. However, the large predicted-versus-oracle routing gap showed that the hierarchy retained substantial conditional predictive capability and that routing error was a major end-to-end bottleneck. Secondary task comparisons suggest possible shared-representation trade-offs, while rare-class and external-generalization limitations remain unresolved.

## Mandatory claim guardrails carried into manuscript revision

Do not claim:

- state-of-the-art performance;
- clinical superiority;
- clinical readiness, safety, or utility;
- external population or multi-centre generalization;
- causal negative transfer;
- reliable SCC-specific superiority;
- stable rare Task 3 T-category conclusions;
- oracle-gate macro-F1 as deployable performance;
- routing as the sole cause of hierarchical underperformance;
- deployment-speed superiority based only on isolated forward-pass benchmarks.

McNemar must only be used to support paired correctness/accuracy discordance, not macro-F1 significance.

## Gate 06A edit map

### Retain with verification

- repository LaTeX structure;
- IEEE double-blind wrapper if still required by target venue;
- dataset mapping and leakage-aware split description;
- much of preprocessing description;
- core explanation of hard-routing error propagation;
- bootstrap and McNemar methodology, with corrected interpretation;
- several existing foundational references;
- core external-validation and rare-class limitations.

### Revise substantially

- title if needed for exact final-system framing;
- introduction;
- related work positioning;
- problem formulation;
- methodology architecture/system description;
- evaluation-protocol wording;
- limitations;
- contribution statements.

### Replace

- abstract results paragraph;
- primary results tables;
- paired-comparison results;
- classwise result table;
- routing numbers;
- results interpretation;
- discussion;
- conclusion;
- current final-system figures that depict the obsolete architecture/results.

### Add

- Phase 05 canonical figures;
- shared-vs-standalone task evidence;
- storage/performance trade-off discussion with explicit runtime caveat;
- test-consumption/frozen-evaluation guardrail;
- SCC CI qualifier;
- Task 3 `7/2/1` rare-support qualifier;
- possible-negative-transfer non-causal qualifier.

## Gate 06A conclusion

The repository contains a complete editable manuscript, so Phase 06 can proceed from Git rather than relying on an Overleaf-only source. The manuscript is scientifically coherent for the earlier ICCIT-era experiment, but it is **not submission-ready for the Phase 05 final evidence** because the model identities, headline results, statistical conclusion, routing decomposition, figures, discussion, and conclusion are materially outdated.

No contradiction was found in the Phase 05 frozen evidence itself; the contradiction is between that evidence and the older manuscript narrative.

## Exact next gate

**Gate 06B — Results and Evaluation Integration**

Gate 06B should update the manuscript with the canonical Phase 05:

- main result table;
- paired confidence intervals;
- exact McNemar result;
- classwise evidence;
- routing decomposition;
- oracle-routing diagnostic;
- shared-vs-standalone evidence;

while preserving all frozen-test and claim-strength guardrails.

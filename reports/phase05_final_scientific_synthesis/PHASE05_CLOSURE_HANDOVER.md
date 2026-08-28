# Final Year Project — Phase 05 Closure and Handover

Date: 2026-08-28

## Phase status

**PHASE 05 — FINAL SCIENTIFIC SYNTHESIS AND PUBLICATION-READY EVIDENCE: CLOSED / PASS**

Phase 05 was reporting-, synthesis-, and publication-packaging-only. No training, inference rerun, checkpoint replacement, preprocessing modification, threshold tuning, recalibration, model reselection, or reuse of the consumed internal test occurred.

## Repository

- Repository: `1sazzad/Skin-Cancer-Hierarchical-Classification`
- Authoritative local workspace: `F:\Research\Final Year\Skin-Cancer-Hierarchical-Classification-main`
- Phase 05 branch: `phase05-final-scientific-synthesis`
- Phase 04 closure commit used as frozen scientific starting point: `6c1d07bcd2d6bf93ce756340155cdc23f3eef01b`

## Phase 05 key commits

- Initial evidence inventory: `c63e0cfebd8305498ba19d4a8172f23692d6e5e5`
- Publication evidence map: `5b97d66031739f079be744f81e6df60996f1b42f`
- Publication-ready tables: `3f4273e9e0d0e4e8caae6dfa76936c97c9a4c454`
- Figure specification: `e49e363d1a6d15241e2328723788dd5fd5f73ea0`
- Figure-generation script: `e0924e47bf218823020e5a8f9ba2075b1dedda1b`
- Publication figures: `787a9fe`
- Gate 05C synthesis: `007ea6b59ba64498f053bd032622581889404602`
- Gate 05D evidence-package audit: `ca4a086ab2a019f635411b0c256705d4eb3dd8d6`

## Completed gates

### Gate 05A — Evidence Inventory and Consistency Audit

**PASS**

- Frozen Phase 04 artifacts inventoried.
- Canonical numbers reconciled.
- Claim-strength boundaries documented.
- Publication claim-to-evidence mapping established.

### Gate 05B — Publication-Ready Tables and Figures

**PASS**

- Canonical manuscript tables prepared.
- Four publication figures generated and visually/technically verified.
- Figure-generation script retained for reproducibility.
- Figures stored under the authoritative project workspace and committed.

### Gate 05C — Final Scientific Synthesis

**PASS**

Publication-ready sections completed:

- Results
- Discussion
- Limitations
- Contribution statements
- Conclusion

All synthesis text was validated against the frozen Phase 04 evidence and approved claim guardrails.

### Gate 05D — Publication-Ready Evidence Package and Claim Audit

**PASS**

- Tables, figures, synthesis, captions, statistical wording, and limitations cross-audited.
- No numerical contradiction identified.
- No unsupported headline claim identified.
- Mandatory publication qualifiers locked.

### Gate 05E — Phase Closure and Handover

**PASS**

This document closes Phase 05.

## Authoritative Phase 05 artifacts

All Phase 05 scientific artifacts are under:

`reports/phase05_final_scientific_synthesis/`

Key files:

- `gate05a_evidence_inventory.md`
- `gate05a_publication_evidence_map.md`
- `gate05b_publication_tables.md`
- `gate05b_figure_specification.md`
- `gate05c_final_scientific_synthesis.md`
- `gate05d_publication_evidence_package.md`
- `figures/fig1_end_to_end_macro_f1.png`
- `figures/fig2_routing_loss.png`
- `figures/fig3_classwise_f1.png`
- `figures/fig4_classwise_delta_ci.png`

Reproducible figure script:

`scripts/reporting/phase05_make_publication_figures.py`

## Frozen internal-test evidence

Internal-test sample count:

`N = 3668`

### Main end-to-end results

- Flat four-class macro-F1: `0.6192224685168973`
- Shared predicted-gate hierarchy macro-F1: `0.5685909456725847`
- Shared oracle-gate hierarchy macro-F1: `0.7769758578867025`
- Routing loss: `0.2083849122141178`

### Paired overall comparison

Hierarchy minus flat macro-F1:

- Delta: `-0.050631522844312604`
- 95% CI: `[-0.07599324143900181, -0.02482655293932654]`

Hierarchy minus flat accuracy:

- Delta: `-0.05452562704471098`
- 95% CI: `[-0.06870229007633588, -0.040348964013086075]`

McNemar:

- Hierarchy-only correct: `274`
- Flat-only correct: `474`
- Exact two-sided p: `2.49117637964427e-13`

## Final scientific conclusions

1. The flat four-class classifier is the stronger **deployed** system under the frozen internal-test protocol.

2. The deployed shared hierarchy underperformed the flat model in both macro-F1 and accuracy, and the paired confidence intervals for the overall differences excluded zero.

3. Oracle routing raised hierarchical macro-F1 from `0.568591` to `0.776976`, producing a routing-associated gap of `0.208385`.

4. Routing error is therefore a major source of end-to-end hierarchical degradation, but is not established as the sole source.

5. Conditional Task 2 performance remained comparatively strong on the true malignant subset, indicating that downstream subtype capability was not fully realized in the deployed hard-routing pipeline.

6. Shared Task 1 and Task 2 underperformed their standalone counterparts. This is consistent with possible negative transfer or insufficient task-specific specialization, but causation is not established.

7. SCC point estimates favored the flat model, but SCC support was only `94` and the paired confidence interval crossed zero. No reliable SCC-specific superiority is claimed.

8. Task 3 remains highly unstable because T2/T3/T4 supports are only `7/2/1`. Strong rare-T-category conclusions are prohibited.

9. Validation and internal-test evidence agreed in the primary direction, supporting a within-study conclusion but not external or clinical generalization.

10. The shared architecture has a storage advantage over three independently stored task models, but isolated single-forward-pass efficiency evidence does not establish superior full conditional deployment latency.

## Publication framing lock

The paper must be framed as a **controlled comparative and failure-analysis study**, not as a leaderboard-performance or state-of-the-art paper.

Strongest approved paper message:

> Under a frozen internal evaluation protocol, the flat four-class classifier provided stronger deployable performance than the shared hard-routing hierarchy. However, the large predicted-versus-oracle routing gap showed that the hierarchy retained substantial conditional predictive capability and that routing error was a major end-to-end bottleneck. Secondary task comparisons suggest possible shared-representation trade-offs, while rare-class and external-generalization limitations remain unresolved.

## Mandatory claim guardrails

Do not claim:

- state-of-the-art performance;
- clinical superiority, readiness, safety, or clinical utility;
- external population or multi-centre generalization;
- causal negative transfer;
- reliable SCC-specific superiority;
- stable Task 3 T2/T3/T4 conclusions;
- oracle-gate macro-F1 as deployable performance;
- routing as the sole cause of hierarchical underperformance;
- full deployment-speed superiority based only on isolated forward-pass benchmarks.

## Frozen-test rule carried forward

The internal test has already been consumed once.

No future phase may use its results to perform:

- threshold tuning;
- checkpoint selection;
- architecture selection;
- preprocessing modification;
- calibration or post-hoc optimization;
- model retraining or reselection targeted to the test findings.

Any future experimental improvement must use training/validation evidence only and require an appropriately separate evaluation plan.

## Repository/worktree note

The Phase 05 GitHub branch was confirmed at Gate 05D to point to commit `ca4a086ab2a019f635411b0c256705d4eb3dd8d6` before this closure commit.

The local workspace previously contained two pre-existing untracked items:

- `.git.broken-worktree-pointer`
- `backups/`

They were intentionally not destructively inspected, modified, staged, or committed during Phase 05. Their existence is a local repository-hygiene issue only and does not affect the locked Phase 05 scientific evidence.

## Unresolved scientific issues

These remain research limitations or future-work directions rather than Phase 05 blockers:

1. Robust routing without hard error propagation.
2. Possible negative transfer in the shared representation.
3. Extremely sparse Task 3 advanced T categories.
4. Small SCC support.
5. Lack of external/prospective/multi-centre validation.
6. Lack of full conditional deployment latency benchmarking.
7. No causal study isolating the effect of task sharing.

## Exact next phase

**Phase 06 — Manuscript Integration and Submission-Ready Paper Revision**

Phase 06 must start in a new chat/session and must not introduce new model training automatically.

### Phase 06 objective

Integrate the frozen Phase 05 publication package into the manuscript and produce a submission-ready scientific paper.

### Phase 06 initial scope

1. Audit the current manuscript source before editing.
2. Update the abstract to reflect the final flat-versus-hierarchy and routing findings.
3. Update Methods/Evaluation text to describe the frozen comparative protocol and paired statistics correctly.
4. Replace outdated Results with the Phase 05 canonical evidence.
5. Integrate approved tables and figures with publication-ready captions.
6. Update Discussion and Limitations using the Gate 05C/05D claim boundaries.
7. Reconcile contributions and conclusion with the final scientific story.
8. Audit Related Work and novelty positioning against the final paper framing.
9. Perform a final reviewer-style consistency, citation, page-limit, anonymity, and overclaim audit.

Do not change frozen experimental evidence during manuscript integration.

## Final Phase 05 verdict

**CLOSED / PASS**

Phase 05 successfully transformed the frozen comparative evaluation into a publication-ready evidence package with reconciled statistics, reproducible figures, bounded scientific claims, final synthesis, and explicit limitations. The project is ready to move to manuscript integration in a separate Phase 06 session.

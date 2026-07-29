# Phase 08 — ICCIT Manuscript Evidence Map

## Use rule

Current two-stage evidence and future three-stage evidence must remain visibly
separate. “Conditional” below means a claim becomes eligible only if the named
future protocol succeeds; it is not a predicted result.

| Destination | Evidence already available | Evidence still required | Claims forbidden now | Conditionally allowed later |
|---|---|---|---|---|
| Abstract | Locked two-stage flat comparison and uncertainty | Stage 3, final framework, external and efficiency results | Completed three-stage system; superiority; broad generalisation | Exact bounded outcomes from completed protocols |
| Introduction | Original hierarchy and documented research gap | Final contribution statement aligned to actual results | Clinical deployment need as proof of efficacy | Motivation for conditional, partial-label learning |
| Related Work | Project framing only; no repository literature synthesis | Reproducible literature search and cited comparison table | State-of-the-art claim | Scoped novelty claim after literature audit |
| Methodology | ISIC split, Stage 1/2 models, routing, focal loss, paired analysis | EMB mapping, shared design, masked losses, external and XAI protocols | Existing shared three-head model | Implemented design described exactly |
| Experimental Setup | Seed-42 ISIC protocol and Tesla T4 records | Stage 3 split, seeds, HIBA mapping, profiling setup | Equivalent conditions not actually matched | Prospective controls that were followed |
| Results | Phase 03–07 tables and figures | Phase 10–14 outputs | Statistical superiority; Stage 3/external/XAI success | Reported positive, null, or negative results |
| Discussion | Routing loss, SCC weakness, uncertainty limits | Cross-task, domain-shift, XAI, and efficiency interpretation | XAI proves correctness; external test proves generalisation | Bounded interpretation tied to named populations |
| Limitations | Single seed/internal dataset; low SCC support; missing efficiency measures | Remaining dataset-specific limitations | Elimination of leakage or bias | Transparent final limitations |
| Conclusion | Two-stage internal findings only | Final synthesis after Phase 14 | Original proposal completed | Evidence-matched final contribution |
| Tables | `reports/phase07/generated/paper_table_*.csv`; efficiency inventory | Stage 3, shared-versus-separate, external, profiling tables | Invented or mixed-protocol cells | Predeclared final comparisons |
| Figures | Phase 07 architecture, confusion matrices, per-class F1 | Three-stage architecture, external results, preregistered XAI | Current architecture figure as a three-stage implementation | Final-system and representative-case figures |

## Current allowed statements

- The implemented two-stage hierarchy achieved macro-F1 `0.6053674006`, while
  the flat comparator achieved `0.6192224685`, on the locked 3,668-image ISIC
  2019 internal-test population.
- The paired macro-F1 difference was `+0.0138550680`, with paired 95% CI
  `[-0.0142546488, 0.0419633760]`; exact McNemar `p=0.8207415883`.
- No statistically distinguishable macro-F1 difference was established.
- Routing and subtype errors were decomposed; the result is internal,
  single-seed evidence.
- Current efficiency evidence supports parameter/file-size accounting, recorded
  timing with limitations, and conditional-work proxies—not FLOP, memory, or
  speed superiority.

The definitive current claim boundary remains
`reports/phase07/phase07_claims_lock.md`.

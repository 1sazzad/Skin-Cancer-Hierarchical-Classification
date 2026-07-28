# Phase 07 Paper-Table Recommendations

## Six-page ICCIT policy

Use the main architecture comparison and compact per-class F1 table in the main text. Integrate the paired macro-F1 difference, accuracy difference, discordant counts (354 versus 347), and exact McNemar p=0.8207 into the main comparison footnote or adjacent text.

Keep the detailed correctness/agreement and routing-decomposition tables as supporting evidence unless conditional routing is central to the narrative. Do not reduce table fonts below a readable conference-paper standard.

## Table notes

- All systems use the same locked 3,668-sample split.
- Bootstrap intervals are paired and ground-truth-class stratified, with 10,000 replicates and seed 42.
- Per-class comparisons are exploratory; no class-wise p-values or multiplicity-adjusted class-wise inference were produced.
- SCC uncertainty is high because support is 94.
- Routing rows include explicit denominators and must not be described collectively as error categories.

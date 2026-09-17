# Phase 07 Claims Lock

## Supported

- Both architectures were evaluated on the same locked leakage-aware internal-test split.
- Flat observed macro-F1 was 0.619 and hierarchical observed macro-F1 was 0.605.
- The observed flat-minus-hierarchical macro-F1 difference was approximately 0.014.
- The paired 95% confidence interval included zero.
- The analysis did not establish a statistically distinguishable macro-F1 difference.
- Overall paired correctness was similar, and exact McNemar testing did not detect a difference.
- Flat observed balanced accuracy was higher, but its paired interval included zero.
- Per-class findings are exploratory and SCC estimates are uncertain because support was 94.
- Results apply only to the locked ISIC 2019 internal-test split.
- The flat system uses one model decision path per image; the hierarchical design uses conditional routing.

## Carefully qualified

- On this single split, the flat model showed a possible melanoma-specific advantage; this is exploratory, no class-wise p-values were generated, and multiplicity-adjusted class-wise inference was not performed.
- The hierarchical model had slightly higher observed non-malignant and BCC F1; these descriptive differences do not establish superiority.
- Conditional routing changed the observed error distribution on this split; external operational consequences were not evaluated.
- The designs may offer different operational trade-offs, but efficiency and external deployment evidence are not established here.

## Prohibited formulations

- `statistically equivalent`
- `non-inferior`
- `clinically superior`
- `clinically validated`
- `improves diagnosis`
- `reduces mortality`
- `ready for deployment`
- `robust across datasets`
- `generalizes across populations`
- `fair across skin tones`
- `externally validated`
- `causal benefit`
- `statistically significant melanoma advantage`
- `statistically significant SCC advantage`
- `definitive rare-class superiority`
- `claims based on selected individual examples`

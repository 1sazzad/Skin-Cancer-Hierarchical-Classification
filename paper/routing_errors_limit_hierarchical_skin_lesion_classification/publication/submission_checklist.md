# ICCIT 2026 submission checklist

Status date: 2026-07-30

| Check | Status | Evidence / action |
|---|---|---|
| IEEE A4 conference source | PASS | `\documentclass[a4paper,conference]{IEEEtran}` |
| Single PDF | BLOCKED | No `pdflatex`, `latexmk`, or `tectonic` executable was available on PATH; compile before submission |
| Page count <= 6 | PENDING PDF | Must be measured after compilation; source was compacted for six pages but this is not a substitute for measurement |
| Double-blind check | PASS (source) | Anonymous author block; no acknowledgments or self-identifying text |
| Author identity scan | PASS (source) | No names, affiliations, addresses, e-mails, repository URLs, GitHub usernames, institution names, or local paths in `main.tex` |
| References resolved | PASS (source) / PENDING PDF | Every citation key has a BibTeX entry; final log must confirm |
| Figures/tables count | PASS | Exactly 2 figures and 3 tables; all are referenced |
| Claims audit complete | PASS | Every major numerical statement is recorded in `claims_traceability.md` |
| Limitations included | PASS | Section V covers split, patient-ID, external, statistical, rare-class, and standalone Stage-3 limits |
| External evaluation not claimed | PASS | Explicitly stated as not completed |
| XAI not claimed | PASS | Explainability explicitly listed as unevaluated |
| Integrated Stage-3 not claimed | PASS | Stage-3 is explicitly standalone and not integrated |
| Clinical deployment readiness not claimed | PASS | Explicitly disclaimed |
| Statistical superiority/equivalence not claimed | PASS | Wording is “not statistically distinguishable”; equivalence and non-inferiority are disclaimed |
| HIBA excluded as result | PASS | HIBA does not appear in the manuscript |
| Consistent class order | PASS | non-malignant, melanoma, BCC, SCC |
| TODO/FIXME scan | PASS (source) | None in `main.tex` or `references.bib` |
| Unresolved `??` scan | PASS (source) | None in source; compiled PDF/log remains to check |
| Overfull boxes | PENDING PDF | Inspect compiler log |
| Figures embedded | PENDING PDF | Both vector PDFs exist in `figures/`; verify compiled output |
| Fonts embedded | PENDING PDF | Verify with `pdffonts` or equivalent after compilation |
| `git diff --check` | PASS | Completed with no whitespace errors |

## Mandatory pre-submission commands

From `paper/iccit2026`, compile with the sequence in `README.md`, then run a PDF
page counter, extract PDF text for an identity/`??` scan, inspect the LaTeX log
for unresolved citations/references and material overfull boxes, and verify
embedded fonts. Do not submit until all `PENDING PDF` and `BLOCKED` rows pass.

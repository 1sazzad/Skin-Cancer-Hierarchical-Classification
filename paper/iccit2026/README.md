# ICCIT 2026 manuscript

This directory contains the anonymous, submission-oriented IEEE A4 conference
manuscript built exclusively from the project's locked evidence. It does not
report HIBA as an experimental result, external evaluation, XAI, an integrated
three-stage evaluation, or clinical deployment readiness.

## Files

- `main.tex`: double-blind manuscript source.
- `references.bib`: verified primary-paper bibliography.
- `figures/`: the two locked vector figures copied from Phase 07.
- `claims_traceability.md`: source audit for numerical claims.
- `submission_checklist.md`: submission validation status.

## Build

With a TeX Live or MiKTeX installation containing `IEEEtran`, run:

```text
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Alternatively, `latexmk -pdf main.tex` may be used. Build from this directory
so the relative figure paths resolve. No LaTeX compiler was available on PATH
when the workspace was created; therefore `main.pdf` must be compiled and its
page count and PDF-level checks completed before submission.

## Scope lock

Do not replace the locked metrics, rerun a model, add author identity before
initial review, or broaden any conclusion beyond `claims_traceability.md`.

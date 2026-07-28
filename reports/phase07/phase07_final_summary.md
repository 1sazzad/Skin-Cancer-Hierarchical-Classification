# Phase 07 Final Summary

## Objective and locked evidence

Phase 07 paired the locked Phase 05 hierarchical and Phase 06C flat stored
predictions, froze and executed a prespecified paired statistical protocol,
independently reviewed the evidence, locked paper claims, audited efficiency,
and produced the ICCIT table and figure package. The analysis used 3,668
paired images with support 2,398 non-malignant, 678 melanoma, 498 BCC, and 94
SCC. Both one-time internal-test protocols remain consumed; reruns are
prohibited.

The pairing gate established identical identifiers and ground truth. The
protocol froze a paired, ground-truth-class-stratified bootstrap with 10,000
replicates, seed 42, fixed labels, and zero-division policy zero. Gate 2A froze
explicit NumPy `method="linear"` percentile quantiles before execution.

## Statistical conclusion

On the locked ISIC 2019 internal-test split, the flat model achieved a higher
observed macro-F1 than the hierarchical model, but the paired 95% bootstrap
confidence interval for the difference included zero. The analysis therefore
did not establish a statistically distinguishable macro-F1 difference.

Macro-F1 was 0.619222 for flat and 0.605367 for hierarchy; the difference was
0.013855 (95% CI −0.014255 to 0.041963). Accuracy was 0.742094 versus
0.740185, with paired difference 0.001908 (95% CI −0.011996 to 0.015812).
The exact McNemar test did not detect a paired-correctness difference
(354 versus 347 discordant cases; p=0.820742). Per-class evidence is
exploratory. SCC uncertainty is high because support is 94.

Routing denominators were independently reconciled. Structural Stage 2
missingness represents Stage-2-not-invoked data availability and coincides
with correctly routed non-malignant cases under the stored union execution
policy; it is not a routing error.

## Claims, efficiency, tables, and figures

Claims are locked in `generated/claims_lock.json`; equivalence,
non-inferiority, clinical superiority, deployment, causal, and external
generalization claims are prohibited. The efficiency audit used static
checkpoint metadata and stored evaluator-loop timing only. Timing is
comparable with limitations, so no speed ratio or faster-system claim is
authorized.

Main paper tables are the architecture comparison and compact per-class F1
comparison. Supporting evidence includes paired correctness, agreement,
routing, and efficiency tables. The final figure package contains:

1. conditional hierarchy and direct flat architecture;
2. side-by-side row-normalized confusion matrices;
3. exploratory per-class F1 with model-specific intervals.

SVG, deterministic PDF, and 600-DPI PNG exports are provided. The optional
paired-correctness figure was omitted as duplicative.

## Reproducibility and validation

```powershell
.\.venv\Scripts\python.exe scripts/run_phase07_statistical_analysis.py --output-directory reports/phase07/generated --control-directory reports/phase07/control --report-path reports/phase07/phase07_statistical_analysis_results.md
.\.venv\Scripts\python.exe scripts/generate_phase07_gate04_evidence.py
.\.venv\Scripts\python.exe scripts/audit_phase07_efficiency_evidence.py
.\.venv\Scripts\python.exe scripts/generate_phase07_paper_figures.py
.\.venv\Scripts\python.exe -m pytest -q
```

Gate 5B targeted tests passed 5/5, all Phase 07 tests passed 97/97, and the
full local suite passed 188/188. Figure generation was repeated with
byte-identical SVG, PDF, PNG, audit, and manifest outputs.

Phase commits: Gate 1 `38c68e12`, Gate 2 `8db8f90b`, Gate 2A `6c5a99a5`,
Gate 3 implementation `8100745a`, Gate 3 results `ed5534c8`, Gate 4
`a5fad87a`, and Gate 5A `5f84d802`.

## Limitations and next phase

Evidence comes from one seed, one internal dataset, and one locked split.
Rare-class uncertainty remains substantial, especially for SCC. No external
validation, equivalence/non-inferiority design, clinical evaluation, fairness
study, prospective deployment test, or definitive latency/memory/FLOP/energy
benchmark is available.

The next phase is independent final branch review and manuscript assembly.
Any push or merge requires explicit human approval; no internal-test rerun is
permitted.

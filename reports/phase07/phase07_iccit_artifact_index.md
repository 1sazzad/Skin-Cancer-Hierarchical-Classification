# Phase 07 ICCIT Artifact Index

| Category | Artifact | Purpose / source | Recommendation | Limitation |
|---|---|---|---|---|
| Main table | `generated/paper_table_main_model_comparison.csv` | Gate 3 point estimates and model-specific intervals | Main text | One locked split |
| Main table | `generated/paper_table_per_class_f1.csv` | Exploratory class F1 and intervals | Main text, compact | No class-wise inference; SCC n=94 |
| Supporting table | `generated/paper_table_correctness_agreement.csv` | Paired correctness and agreement | Supporting evidence | Descriptive except overall McNemar |
| Supporting table | `generated/paper_table_routing_decomposition.csv` | Audited routing counts and denominators | Supporting evidence | Structural missingness is not an error |
| Supporting table | `generated/efficiency_comparison_table.csv` | Static/stored efficiency audit | Supporting evidence | Timing comparable with limitations |
| Main figure | `figures/figure01_architecture.{svg,pdf,png}` | Documented system paths | Required | No clinical-workflow implication |
| Main figure | `figures/figure02_confusion_matrix_comparison.{svg,pdf,png}` | Gate 3 confusion matrices | Required | Descriptive visualization |
| Main figure | `figures/figure03_per_class_f1.{svg,pdf,png}` | Gate 3 model-specific intervals | Required | Exploratory; SCC uncertain |
| Optional figure | Paired correctness figure | Gate 3 paired categories | Omitted | Duplicates table evidence |
| Statistical source | `generated/statistical_analysis_results.json` | Authoritative Gate 3 summary | Cite through tables/text | Internal split only |
| Statistical source | `generated/bootstrap_replicates.csv` | All 10,000 frozen replicates | Reproducibility only | Must not be modified |
| Claims | `generated/claims_lock.json` | Supported, qualified, prohibited claims | Mandatory review source | No claim expansion |
| Efficiency | `generated/efficiency_claims_lock.json` | Efficiency wording boundary | Mandatory review source | No speedup claim |
| Figure audit | `generated/figure_data_audit.json` | Values, sources, normalization, hashes | Reproducibility | Environment uses Matplotlib 3.10.8 |
| Figure manifest | `generated/figure_artifact_manifest.txt` | Figure artifact hashes | Reproducibility | Regenerate with fixed command |
| Reproducibility | `scripts/generate_phase07_paper_figures.py` | Deterministic exports | Retain | Requires project Python environment |

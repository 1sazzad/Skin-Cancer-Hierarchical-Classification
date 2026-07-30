# Claims traceability

All entries below were manually checked against locked, repository-resident
evidence on 2026-07-30. `Verified` means the manuscript value agrees with the
named source; it does not imply external replication.

| Manuscript section | Exact claim | Source file | Source field/table | Status |
|---|---|---|---|---|
| Abstract, III-A | Internal test has 3,668 images | `reports/phase05/conditional_hierarchical_internal_evaluation.md` | Experimental status; headline results | Verified |
| III-A, Table I | Train/validation/test totals are 17,124/3,668/3,668 | `reports/dataset_audits/isic2019_phase02_class_statistics_seed42.csv` | `split_total` by split | Verified |
| III-A, Table I | Four-class train/validation/test class counts | `reports/dataset_audits/isic2019_phase02_class_statistics_seed42.csv` | Stage-1 non-malignant plus Stage-2 malignant subtype rows | Verified |
| III-A | Split groups connected lesion IDs and exact hashes; patient IDs unavailable | `reports/dataset_audits/isic2019_split_group_audit.json` | `connected_component_grouping.strategy`; `recommended_split_policy` | Verified |
| III-B/C | EfficientNet-B0, ImageNet initialization, seed 42, batch 64, AdamW, LR 0.0003, weight decay 0.0001, cosine annealing, 30 epochs, patience 7 | `configs/experiments/phase03_stage01_isic2019_efficientnet_b0_cross_entropy.yaml`; `configs/experiments/phase04_stage02_isic2019_efficientnet_b0_class_balanced_focal_loss.yaml`; `configs/experiments/phase06_flat_four_class_isic2019_efficientnet_b0_cross_entropy.yaml` | `model`, `loader`, and `training` fields | Verified |
| III-C | Stage-2 class-balanced focal loss uses beta 0.9999 and gamma 2 | `reports/phase05/conditional_hierarchical_internal_evaluation.md` | Frozen model provenance, Stage 2 | Verified |
| III-D | Paired, ground-truth-stratified percentile bootstrap uses 10,000 replicates, seed 42, 95% confidence | `configs/analysis/phase07_paired_model_comparison.yaml` | `statistical_protocol.bootstrap` | Verified |
| IV-A, Table II | Hierarchical accuracy 0.740185 and macro-F1 0.605367 | `reports/phase05/conditional_hierarchical_internal_evaluation.md` | Headline results, predicted-gate end-to-end | Verified |
| IV-A, Table II | Flat accuracy 0.742094 and macro-F1 0.619222 | `reports/phase06/phase06c_selected_flat_internal_test_result.md` | Internal-test results | Verified |
| IV-A | Flat-minus-hierarchical macro-F1 is +0.013855, 95% CI [-0.014255, 0.041963] | `reports/phase07/generated/paper_table_paired_comparison.csv` | Macro-F1 row | Verified |
| IV-A | Accuracy difference 0.001908, 95% CI [-0.011996, 0.015812] | `reports/phase07/generated/paper_table_paired_comparison.csv` | Accuracy row | Verified |
| IV-A | Flat-only/hierarchy-only correct counts are 354/347; exact McNemar p=0.820742 | `reports/phase07/generated/paper_table_correctness_agreement.csv` | `flat_only_correct`, `hierarchy_only_correct`, `exact_mcnemar_p_value` | Verified |
| IV-B, Table III | Oracle-gate macro-F1 0.793656 and routing-related loss 0.188289 | `reports/phase05/conditional_hierarchical_internal_evaluation.md` | Error propagation table | Verified |
| IV-B, Table III | Malignant blocking is 255/1,270 = 20.079% | `reports/phase07/generated/paper_table_routing_decomposition.csv` | `true_malignant_routed_non_malignant` | Verified |
| IV-B, Table III | Incorrect benign routing is 529/2,398 = 22.060% | `reports/phase07/generated/paper_table_routing_decomposition.csv` | `true_non_malignant_routed_stage2` | Verified |
| IV-B | Wrong subtype after correct routing is 169/1,015 = 16.650% | `reports/phase07/generated/paper_table_routing_decomposition.csv` | `correct_malignant_route_wrong_subtype` | Verified |
| IV-C | Per-class flat/hierarchical F1 and supports | `reports/phase07/generated/paper_table_per_class_f1.csv` | All class rows | Verified |
| IV-D | Stage-3 cohort 848; split 594/127/127 | `reports/phase09/isic_stage03_fasttrack_result.md` | Dataset audit; deterministic leakage-safe split | Verified |
| IV-D, Table III | Weighted-CE Stage-3 macro-F1 0.275611 and balanced accuracy 0.386039 | `reports/phase09/isic_stage03_fasttrack_result.md` | Locked internal-test comparison | Verified |
| IV-D, Table III | Stage-3 T2 and T4 recall are zero, with test supports 7 and 1 | `reports/phase09/isic_stage03_fasttrack_result.md` | Per-class test comparison | Verified |
| V | No external evaluation, patient independence guarantee, XAI, fairness, calibration, or clinical deployment evidence | `reports/phase07/phase07_claims_lock.md`; `reports/phase09/isic_stage03_fasttrack_result.md`; `reports/dataset_audits/isic2019_split_group_audit.json` | Prohibited formulations; limitations | Verified |

## Reference verification

Bibliographic metadata was checked against official publisher/proceedings pages
or primary-paper records. No repository URL, user name, e-mail address, author
identity, or institution-identifying project text appears in `main.tex`.

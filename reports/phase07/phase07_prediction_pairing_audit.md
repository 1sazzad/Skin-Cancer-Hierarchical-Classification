# Phase 07 Prediction Pairing Audit

## Gate result

The first Phase 07 gate passed. The locked Phase 05 hierarchical and Phase 06C
flat predictions form an exact, deterministic one-to-one pairing of 3,668
`image_id` values. Pairing uses exact identifier equality followed by ascending
lexical sort and is independent of source CSV row order.

No statistical analysis was executed.

## Authoritative artifacts

The Phase 05 artifact is:

`runs/phase05_hierarchical_internal_test/locked_primary_evaluation/per_image_hierarchical_predictions.csv`

- Size: 1,458,137 bytes
- SHA-256: `391557deb9a1aeb9b9f97edc9d3d38759e597d56b54bfdbab9ea7482451a221a`
- Prediction rows: 3,668
- Provenance: its hash matches the 18-entry locked evaluation manifest,
  `runs/phase05_hierarchical_internal_test/locked_primary_evaluation_artifact_sha256.txt`.

The Phase 06C artifact is canonically:

`runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/internal_test_predictions.csv`

- Size: 1,136,735 bytes
- SHA-256: `08b3462549210ed7f2330a687c37a6de4e013e00185fadc3167aa980995e497d`
- Prediction rows: 3,668
- Provenance: the canonical run directory is artifact-managed and absent from
  this checkout. The file was read from the documented verified local backup,
  `runs/backups/phase06c/phase06c_selected_flat_internal_test_550e7cdb1144.tar.gz`.
  The archive SHA-256 is
  `b76762b53a35a8d9b0aa96621d78ea0e4421aa6e8052d068ffc10648a4e63e91`,
  and every file matches the archive's 12-entry embedded artifact manifest.

The other inventory entries are historical Phase 03 Stage 1, Phase 03 Stage 2,
and Phase 04 Stage 2 prediction artifacts. They are not selected because they
are binary-stage or malignant-subtype evaluations rather than either locked
four-class comparison endpoint. No duplicate copy of either selected CSV was
found.

## Schema and mapping

Both files use `image_id` as the stable identifier.

For Phase 05, ground truth is `final_target_label` / `final_target_index`, and
the comparison prediction is `predicted_gate_predicted_label` /
`predicted_gate_predicted_index`. For Phase 06C, ground truth is
`target_label` / `target_index`, and the prediction is `predicted_label` /
`predicted_index`.

The common class order and index mapping are:

| Index | Class |
|---:|---|
| 0 | `non_malignant` |
| 1 | `melanoma` |
| 2 | `bcc` |
| 3 | `scc` |

Normalization was explicitly limited to trimming, lowercasing, and converting
hyphens or whitespace to underscores. Stored four-class labels already use the
normalized spellings. Label/index agreement was checked in every row.

Ground-truth support is identical:

| Class | Support |
|---|---:|
| `non_malignant` | 2,398 |
| `melanoma` | 678 |
| `bcc` | 498 |
| `scc` | 94 |

## Integrity and identity findings

- Duplicate columns: none.
- Empty or malformed CSV rows: none.
- Missing `image_id` values: none.
- Duplicate `image_id` values: none.
- Identifier sets: identical; no samples occur in only one file.
- Ground-truth disagreements: none.
- Unsupported target or endpoint-prediction labels: none.
- Unexpected audited categorical values: none.
- The identical identifiers, labels, support counts, file hashes, and locked
  protocol provenance establish that rows represent the same frozen seed-42
  internal-test split.

The Phase 05 file also stores image paths, split groups, source-file hashes,
Stage 1 targets/predictions/correctness/probabilities, Stage 2 execution,
targets/predictions/correctness/probabilities, oracle-gate predictions and
correctness, predicted-gate correctness, and routing status. Stage 2 values are
structurally absent when not applicable: target and correctness fields are
empty for 2,398 non-malignant samples; prediction and probability fields are
empty for 1,869 samples on which Stage 2 was not executed. These are expected
conditional-schema values, not malformed data.

Observed Phase 05 routing values are `correctly_not_routed`,
`correctly_routed_subtype_correct`, `correctly_routed_subtype_error`,
`malignant_blocked_by_stage_1`, and `non_malignant_incorrectly_routed`.
Boolean/correctness fields use `0` and `1`, with empty Stage 2 correctness only
where not applicable. Stage labels use the documented binary Stage 1 and
three-class Stage 2 definitions.

The Phase 06C file also stores image paths, split groups, source-file hashes,
correctness, and four per-class probability columns. It has no missing values;
correctness values are `0` or `1`.

No probability, routing, stage-prediction, correctness, label, or source row
was copied back into or modified within a locked directory. The paired
manifest intentionally contains only identity, ground truth, and the two
endpoint predictions.

## Generated evidence

Machine-readable results are under `reports/phase07/generated/`:

- `prediction_file_inventory.csv`
- `selected_prediction_artifacts.json`
- `prediction_pairing_audit.json`
- `paired_prediction_manifest.csv`

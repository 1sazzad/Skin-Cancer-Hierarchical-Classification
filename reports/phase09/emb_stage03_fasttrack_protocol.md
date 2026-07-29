# EMB Stage-3 Fast-Track Protocol

## Scope and dataset identity

This emergency ICCIT experiment is a standalone five-class melanoma severity
classifier, not the future shared three-task model. The dataset is the Early
Stage Melanoma benchmark (EMB), acquired only from the official
`https://github.com/Oichii/EMB.git` repository and the image sources named by
its README. The exact source commit is recorded at acquisition.

## Labels and modality

The official `stage_ajcc` field is authoritative: `0 -> Tis`, `1 -> T1`,
`2 -> T2`, `3 -> T3`, and `4 -> T4`. Thickness is never used to manufacture a
label; it is only checked for reporting consistency. Primary training includes
dermoscopic images only. Clinical photographs are excluded to align with the
existing dermoscopic project models.

## VM-only acquisition and audit gate

EMB repository/data acquisition, metadata and image audit, duplicate/overlap
audit, real split generation, sanity training, full training, inference, and
evaluation occur only on the Azure Tesla T4 VM. Raw files live under
`data/raw/emb/`, the source clone under ignored `data/external/`, and neither
images nor archives are committed. Absence of identifiable usage/licence terms
is an immediate NO-GO until written authorization is recorded.

The audit records row and identifier counts, stage/modality/source counts,
missing and corrupt images, SHA-256 duplicate groups and label conflicts, ISIC
2019 identifier/hash overlap, thickness consistency, grouping fields, and a
GO/NO-GO verdict. Any missing/unreadable image or conflicting duplicate label
blocks splitting.

## Split protocol

The seed-42 primary split is dermoscopic-only, 70% train, 15% validation, and
15% untouched test, stratified by T-category. A valid complete patient
identifier is preferred, then a valid complete lesion identifier. Identifiers
are never invented. If neither exists, the limitation is recorded and exact
hash groups remain together. Image IDs, groups, and hashes may not cross
splits. All five classes must remain in train and, where mathematically
possible, validation/test. Optional inverse-frequency weights are computed
from the training partition only.

## Model and selection

EfficientNet-B0 uses ImageNet initialization, five logits in fixed
`[Tis, T1, T2, T3, T4]` order, cross entropy, AdamW, seed 42, CUDA mixed
precision, at most 30 epochs, and early stopping. The best checkpoint is chosen
only by validation macro-F1. The existing runner writes resolved configuration,
environment, history, checkpoints, and validation metrics. The frozen test
split is accessed once, only after model selection, by the existing evaluator,
which writes predictions and metrics.

## Limitations

EMB licensing/usage permission must be established before image acquisition.
Grouping independence depends on identifiers actually supplied by official
metadata. Source/modality column naming is validated during the VM audit.
Thickness consistency checks cannot replace the official stage field.

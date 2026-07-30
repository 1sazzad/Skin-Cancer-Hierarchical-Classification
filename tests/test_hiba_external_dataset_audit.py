from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


SCRIPT = Path("scripts/audit_hiba_external_dataset.py")
spec = importlib.util.spec_from_file_location("hiba_audit", SCRIPT)
assert spec is not None and spec.loader is not None
hiba_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hiba_audit)


def _write_csv(path: Path, rows: list[dict[str, str]], *, bom: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig" if bom else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mapping(path: Path) -> None:
    path.write_text(
        """
licence_policy:
  accepted_values:
    - CC-BY
mappings:
  melanoma:
    canonical_diagnosis: melanoma
    mapped_final_label: melanoma
    mapped_stage_1_label: malignant
    mapped_stage_2_label: melanoma
    include_primary_evaluation: true
    exclusion_reason: ""
  basal cell carcinoma:
    canonical_diagnosis: basal_cell_carcinoma
    mapped_final_label: bcc
    mapped_stage_1_label: malignant
    mapped_stage_2_label: bcc
    include_primary_evaluation: true
    exclusion_reason: ""
  squamous cell carcinoma:
    canonical_diagnosis: squamous_cell_carcinoma
    mapped_final_label: scc
    mapped_stage_1_label: malignant
    mapped_stage_2_label: scc
    include_primary_evaluation: true
    exclusion_reason: ""
  actinic keratosis:
    canonical_diagnosis: actinic_keratosis
    mapped_final_label: ""
    mapped_stage_1_label: ""
    mapped_stage_2_label: ""
    include_primary_evaluation: false
    exclusion_reason: excluded_actinic_keratosis_primary_mapping
""".lstrip(),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    raw = tmp_path / "data" / "external" / "hiba"
    (raw / "images").mkdir(parents=True)
    (raw / "images" / "one.jpg").write_bytes(b"first-image")
    (raw / "images" / "two.jpg").write_bytes(b"second-image")
    (raw / "images" / "three.jpg").write_bytes(b"third-image")
    (raw / "images" / "four.jpg").write_bytes(b"fourth-image")
    mapping = tmp_path / "mapping.yaml"
    _mapping(mapping)
    isic = tmp_path / "isic.csv"
    _write_csv(isic, [{"image_id": "ISIC_other", "file_sha256": "a" * 64}])
    metadata = tmp_path / "metadata.csv"
    return raw, mapping, isic, metadata


def _row(**changes: str) -> dict[str, str]:
    row = {
        "image_id": "HIBA_1",
        "image_path": "images/one.jpg",
        "patient_id": "",
        "lesion_id": "",
        "diagnosis": " Melanoma ",
        "modality": "Dermoscopy",
        "attribution": "HIBA contributors",
        "license": "CC-BY",
        "source_reference": "https://doi.org/10.34970/587329",
    }
    row.update(changes)
    return row


def _audit(tmp_path: Path, rows: list[dict[str, str]], *, bom: bool = False):
    raw, mapping, isic, metadata = _inputs(tmp_path)
    _write_csv(metadata, rows, bom=bom)
    return hiba_audit.audit_dataset(
        metadata, raw, mapping, isic, project_root=tmp_path
    )


def test_protocol_configs_freeze_required_boundaries() -> None:
    mapping = yaml.safe_load(
        Path("configs/datasets/hiba_external_label_mapping.yaml").read_text(
            encoding="utf-8"
        )
    )
    evaluation = yaml.safe_load(
        Path("configs/evaluation/phase10_hiba_frozen_zero_shot.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert mapping["matching_policy"]["substring_matching_allowed"] is False
    assert set(mapping["mappings"]) == {
        "melanoma",
        "basal cell carcinoma",
        "squamous cell carcinoma",
        "actinic keratosis",
    }
    assert mapping["mappings"]["actinic keratosis"]["include_primary_evaluation"] is False
    assert mapping["unresolved_entries"][0]["category"] == (
        "benign_diagnosis_vocabulary"
    )
    assert mapping["licence_policy"]["accepted_values"] == ["CC-BY"]
    assert evaluation["protocol_status"] == "protocol_pending_dataset_audit"
    assert evaluation["seed"] == 42
    assert evaluation["preprocessing"]["image_size"] == 224
    assert evaluation["preprocessing"]["resize_shorter_side"] == 256
    assert evaluation["preprocessing"]["normalization"] == "imagenet"
    assert evaluation["decision_policy"]["flat"] == "argmax"
    assert evaluation["decision_policy"]["external_calibration_fitting_allowed"] is False
    assert evaluation["data"]["identical_manifest_for_flat_and_hierarchy"] is True
    assert evaluation["execution"]["inference_authorized"] is False
    assert evaluation["execution"]["refuse_existing_output"] is True


def test_valid_dermoscopic_row_preserves_original_and_missing_ids(tmp_path: Path) -> None:
    manifest, audit, checksums = _audit(tmp_path, [_row()], bom=True)
    assert manifest[0]["original_diagnosis"] == " Melanoma "
    assert manifest[0]["patient_id"] == ""
    assert manifest[0]["lesion_id"] == ""
    assert manifest[0]["mapped_final_label"] == "melanoma"
    assert manifest[0]["include_primary_evaluation"] == "true"
    assert audit["approval"]["automated_gates_passed"] is False
    assert audit["approval"]["approved_for_primary_evaluation"] is False
    assert audit["class_support_gate"]["approved_primary_support"] == {
        "bcc": 0,
        "melanoma": 1,
        "non_malignant": 0,
        "scc": 0,
    }
    assert checksums[0].startswith(hashlib.sha256(b"first-image").hexdigest())


def test_unknown_mapping_fails_closed_without_substring_guess(tmp_path: Path) -> None:
    manifest, audit, _ = _audit(
        tmp_path, [_row(diagnosis="possible melanoma / uncertain")]
    )
    assert manifest[0]["original_diagnosis"] == "possible melanoma / uncertain"
    assert manifest[0]["mapped_final_label"] == ""
    assert manifest[0]["include_primary_evaluation"] == "false"
    assert manifest[0]["exclusion_reason"] == "unresolved_diagnosis_mapping"
    assert audit["unresolved_original_diagnoses"] == [
        "possible melanoma / uncertain"
    ]
    assert "unresolved_diagnosis_mappings" in audit["approval"]["blockers"]


def test_guessed_benign_label_remains_unresolved(tmp_path: Path) -> None:
    manifest, audit, _ = _audit(
        tmp_path, [_row(diagnosis="melanocytic nevus")]
    )
    assert manifest[0]["canonical_diagnosis"] == ""
    assert manifest[0]["mapped_final_label"] == ""
    assert manifest[0]["exclusion_reason"] == "unresolved_diagnosis_mapping"
    assert audit["unresolved_original_diagnoses"] == ["melanocytic nevus"]


def test_cc_by_passes_but_cc_by_4_remains_unsupported(
    tmp_path: Path,
) -> None:
    accepted_manifest, _, _ = _audit(tmp_path / "accepted", [_row()])
    assert accepted_manifest[0]["license"] == "CC-BY"
    assert accepted_manifest[0]["include_primary_evaluation"] == "true"

    manifest, audit, _ = _audit(
        tmp_path / "unsupported", [_row(license="CC BY 4.0")]
    )
    assert manifest[0]["license"] == "CC BY 4.0"
    assert manifest[0]["include_primary_evaluation"] == "false"
    assert manifest[0]["exclusion_reason"] == (
        "excluded_unsupported_or_unknown_licence"
    )
    assert "licence_attribution_or_source_gate_failed" in (
        audit["approval"]["blockers"]
    )


def test_zero_approved_rows_blocks_approval(tmp_path: Path) -> None:
    _, audit, _ = _audit(tmp_path, [_row(diagnosis="actinic keratosis")])
    assert audit["counts"]["approved_primary_rows"] == 0
    assert "zero_approved_primary_rows" in audit["approval"]["blockers"]


def test_absent_mapped_class_support_blocks_pending_human_review(
    tmp_path: Path,
) -> None:
    _, audit, _ = _audit(tmp_path, [_row()])
    assert audit["class_support_gate"]["absent_classes"] == [
        "bcc", "non_malignant", "scc"
    ]
    assert "absent_mapped_classes_require_human_feasibility_review" in (
        audit["approval"]["blockers"]
    )
    assert audit["class_support_gate"]["numeric_minimum_support_threshold"] is None


def test_clinical_and_actinic_keratosis_rows_are_explicitly_excluded(
    tmp_path: Path,
) -> None:
    rows = [
        _row(modality="clinical"),
        _row(
            image_id="HIBA_2", image_path="images/two.jpg",
            diagnosis="actinic keratosis",
        ),
    ]
    manifest, _, _ = _audit(tmp_path, rows)
    assert manifest[0]["exclusion_reason"] == "excluded_non_dermoscopic_clinical"
    assert manifest[1]["exclusion_reason"] == (
        "excluded_actinic_keratosis_primary_mapping"
    )
    assert all(row["include_primary_evaluation"] == "false" for row in manifest)


def test_duplicate_image_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(hiba_audit.AuditError, match="Duplicate image_id"):
        _audit(tmp_path, [_row(), _row()])


def test_missing_metadata_column_is_rejected(tmp_path: Path) -> None:
    row = _row()
    del row["license"]
    with pytest.raises(hiba_audit.AuditError, match="license"):
        _audit(tmp_path, [row])


@pytest.mark.parametrize(
    "entry,match",
    [
        (
            """
    canonical_diagnosis: melanoma
    mapped_final_label: melanoma
    mapped_stage_1_label: malignant
    mapped_stage_2_label: melanoma
    exclusion_reason: ""
""",
            "missing keys",
        ),
        (
            """
    canonical_diagnosis: melanoma
    mapped_final_label: melanoma
    mapped_stage_1_label: malignant
    mapped_stage_2_label: melanoma
    include_primary_evaluation: "true"
    exclusion_reason: ""
""",
            "must be boolean",
        ),
        (
            """
    canonical_diagnosis: melanoma
    mapped_final_label: melanoma
    mapped_stage_1_label: non_malignant
    mapped_stage_2_label: ""
    include_primary_evaluation: true
    exclusion_reason: ""
""",
            "inconsistent stage labels",
        ),
    ],
)
def test_invalid_mapping_schemas_are_rejected(
    tmp_path: Path, entry: str, match: str
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "licence_policy:\n"
        "  accepted_values: [CC-BY]\n"
        "mappings:\n"
        "  melanoma:\n"
        + entry,
        encoding="utf-8",
    )
    with pytest.raises(hiba_audit.AuditError, match=match):
        hiba_audit.load_mapping(path)


def test_excluded_mapping_with_blank_reason_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """
licence_policy:
  accepted_values: [CC-BY]
mappings:
  actinic keratosis:
    canonical_diagnosis: actinic_keratosis
    mapped_final_label: ""
    mapped_stage_1_label: ""
    mapped_stage_2_label: ""
    include_primary_evaluation: false
    exclusion_reason: ""
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(hiba_audit.AuditError, match="requires an exclusion reason"):
        hiba_audit.load_mapping(path)


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    with pytest.raises(hiba_audit.AuditError, match="escapes"):
        _audit(tmp_path, [_row(image_path="../../../outside.jpg")])


def test_original_canonical_and_final_conflicts_are_reported_separately(
    tmp_path: Path,
) -> None:
    raw, mapping, isic, metadata = _inputs(tmp_path)
    (raw / "images" / "two.jpg").write_bytes(b"first-image")
    _write_csv(
        metadata,
        [
            _row(lesion_id="L1"),
            _row(
                image_id="HIBA_2", image_path="images/two.jpg", lesion_id="L1",
                diagnosis="actinic keratosis",
            ),
        ],
    )
    _, audit, _ = hiba_audit.audit_dataset(
        metadata, raw, mapping, isic, project_root=tmp_path
    )
    assert audit["integrity"]["exact_duplicate_hash_group_count"] == 1
    for scope in ("identical_hash", "same_lesion"):
        assert len(
            audit["integrity"][f"{scope}_original_diagnosis_conflicts"]
        ) == 1
        assert len(
            audit["integrity"][f"{scope}_canonical_diagnosis_conflicts"]
        ) == 1
        assert len(
            audit["integrity"][f"{scope}_final_label_conflicts"]
        ) == 1


def test_isic_id_or_hash_overlap_blocks_approval(tmp_path: Path) -> None:
    raw, mapping, isic, metadata = _inputs(tmp_path)
    digest = hashlib.sha256(b"first-image").hexdigest()
    _write_csv(isic, [{"image_id": "HIBA_1", "file_sha256": digest}])
    _write_csv(metadata, [_row()])
    _, audit, _ = hiba_audit.audit_dataset(
        metadata, raw, mapping, isic, project_root=tmp_path
    )
    assert audit["isic2019_overlap"]["image_id_overlap_count"] == 1
    assert audit["isic2019_overlap"]["sha256_overlap_count"] == 1
    assert "isic2019_overlap_requires_documented_resolution" in (
        audit["approval"]["blockers"]
    )


def test_main_writes_expected_outputs_atomically(tmp_path: Path) -> None:
    raw, mapping, isic, metadata = _inputs(tmp_path)
    _write_csv(metadata, [_row()])
    manifest = tmp_path / "out" / "manifest.csv"
    audit = tmp_path / "out" / "manifest.audit.json"
    checksums = tmp_path / "out" / "sha256.txt"
    result = hiba_audit.main([
        "--metadata", str(metadata),
        "--raw-root", str(raw),
        "--mapping", str(mapping),
        "--isic-manifest", str(isic),
        "--project-root", str(tmp_path),
        "--manifest-output", str(manifest),
        "--audit-output", str(audit),
        "--checksum-output", str(checksums),
    ])
    assert result == 2
    with manifest.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["dataset"] == "hiba"
    assert json.loads(audit.read_text(encoding="utf-8"))["cohort"][
        "split_created"
    ] is False
    assert checksums.read_text(encoding="utf-8").endswith("\n")


def test_existing_output_is_refused_before_audit(tmp_path: Path) -> None:
    raw, mapping, isic, metadata = _inputs(tmp_path)
    _write_csv(metadata, [_row()])
    manifest = tmp_path / "out" / "manifest.csv"
    manifest.parent.mkdir()
    manifest.write_text("existing evidence\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        hiba_audit.main([
            "--metadata", str(metadata),
            "--raw-root", str(raw),
            "--mapping", str(mapping),
            "--isic-manifest", str(isic),
            "--project-root", str(tmp_path),
            "--manifest-output", str(manifest),
            "--audit-output", str(tmp_path / "out" / "audit.json"),
            "--checksum-output", str(tmp_path / "out" / "sha256.txt"),
        ])
    assert manifest.read_text(encoding="utf-8") == "existing evidence\n"


def test_later_serialization_failure_leaves_no_partial_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = (
        tmp_path / "out" / "manifest.csv",
        tmp_path / "out" / "audit.json",
        tmp_path / "out" / "sha256.txt",
    )
    calls = 0
    original = hiba_audit._serialize_text

    def fail_second_text(path: Path, text: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated checksum serialization failure")
        original(path, text)

    monkeypatch.setattr(hiba_audit, "_serialize_text", fail_second_text)
    with pytest.raises(OSError, match="simulated checksum"):
        hiba_audit.publish_outputs_transactionally(
            *outputs,
            [{"dataset": "hiba"}],
            {"status": "test"},
            ["a" * 64 + "  image.jpg"],
        )
    assert all(not path.exists() for path in outputs)
    assert not list((tmp_path / "out").glob("*.tmp"))

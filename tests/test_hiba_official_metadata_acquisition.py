from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


acquire = _load("hiba_acquire", "scripts/acquire_hiba_official_metadata.py")
inventory = _load("hiba_inventory", "scripts/inventory_hiba_official_metadata.py")


def _collection(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": 251,
        "doi": "10.34970/587329",
        "name": "Hospital Italiano de Buenos Aires - Skin Lesions Images (2019-2022)",
        "imageCount": 1616,
        "attribution": "Exact official attribution",
    }
    payload.update(changes)
    return payload


def _record(index: int, **changes: object) -> dict[str, object]:
    record: dict[str, object] = {
        "isic_id": f"ISIC_{index:07d}",
        "metadata": {
            "clinical": {
                "patient_id": f"P{index // 3}",
                "lesion_id": f"L{index // 2}",
                "diagnosis": "Exact Diagnosis",
                "diagnosis_confirm_type": "histopathology",
            },
            "acquisition": {
                "image_type": "dermoscopic",
                "license": "CC-BY",
                "attribution": "Exact attribution",
                "mime_type": "image/jpeg",
            },
        },
        "public": True,
    }
    record.update(changes)
    return record


def _records() -> list[dict[str, object]]:
    return [_record(index) for index in range(1616)]


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"id": 175}, "Collection ID mismatch"),
        ({"doi": "10.34970/559884"}, "DOI mismatch"),
        ({"name": "Older HIBA release"}, "Title mismatch"),
        ({"imageCount": 1635}, "Expected image-count mismatch"),
    ],
)
def test_collection_identity_mismatches_are_rejected(
    changes: dict[str, object], match: str
) -> None:
    with pytest.raises(acquire.AcquisitionError, match=match):
        acquire.verify_collection(_collection(**changes))


def test_network_access_is_refused_without_explicit_authorization() -> None:
    with pytest.raises(acquire.AcquisitionError, match="network access refused"):
        acquire.collect_metadata()


def test_synthetic_pagination_retrieves_exactly_1616_records() -> None:
    first = [_record(index) for index in range(808)]
    second = [_record(index) for index in range(808, 1616)]
    responses = iter([
        (_collection(), 200),
        ({"results": first, "next": "/api/v2/images/?offset=808"}, 200),
        ({"results": second, "next": None}, 200),
    ])
    urls: list[str] = []

    def fetcher(url: str, **_: object):
        urls.append(url)
        return next(responses)

    collection, records, request_log = acquire.collect_metadata(
        authorize_network=True, fetcher=fetcher
    )
    assert collection["id"] == 251
    assert len(records) == 1616
    assert len(request_log) == 3
    assert request_log[1]["page_number"] == 1
    assert request_log[2]["page_number"] == 2
    assert request_log[2]["item_count"] == 808
    assert urls[0].endswith("/collections/251/")
    assert "collections=251" in urls[1]


def test_duplicate_image_ids_are_rejected() -> None:
    records = _records()
    records[-1] = dict(records[0])
    responses = iter([
        (_collection(), 200),
        ({"results": records, "next": None}, 200),
    ])

    def fetcher(*_: object, **__: object):
        return next(responses)

    with pytest.raises(acquire.AcquisitionError, match="Duplicate ISIC image ID"):
        acquire.collect_metadata(authorize_network=True, fetcher=fetcher)


def test_exactly_1616_records_are_required_for_acquisition() -> None:
    responses = iter([
        (_collection(), 200),
        ({"results": [_record(1)], "next": None}, 200),
    ])

    def fetcher(*_: object, **__: object):
        return next(responses)

    with pytest.raises(acquire.AcquisitionError, match="expected 1616, got 1"):
        acquire.collect_metadata(authorize_network=True, fetcher=fetcher)


def test_later_fixture_page_failure_leaves_no_final_or_temporary_outputs(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixtures"
    fixture.mkdir()
    (fixture / "collection.json").write_text(
        json.dumps(_collection()), encoding="utf-8"
    )
    (fixture / "page_1.json").write_text(
        json.dumps({"results": [_record(1)], "next": "page_2"}), encoding="utf-8"
    )
    (fixture / "page_2.json").write_text("{invalid", encoding="utf-8")
    output = tmp_path / "hiba"
    with pytest.raises(json.JSONDecodeError):
        acquire.main([
            "--fixture-directory", str(fixture),
            "--output-root", str(output),
        ])
    assert not list(output.rglob("*")) if output.exists() else True


def test_existing_acquisition_output_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "hiba"
    existing = root / "source" / "collection_251.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        acquire.publish_acquisition(root, _collection(), _records(), [])
    assert existing.read_text(encoding="utf-8") == "existing\n"


def test_raw_jsonl_preserves_complete_api_objects(tmp_path: Path) -> None:
    records = _records()
    records[0]["unknown_nested"] = {"verbatim": ["A", 2, None]}
    paths = acquire.publish_acquisition(
        tmp_path / "hiba", _collection(), records, []
    )
    with paths["raw_jsonl"].open(encoding="utf-8") as handle:
        restored = [json.loads(line) for line in handle]
    assert restored == records
    assert not (tmp_path / "hiba" / "images").exists()


def test_acquisition_transaction_rolls_back_on_late_serialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "hiba"
    original = acquire._write_jsonl

    def fail_jsonl(path: Path, records: object) -> None:
        original(path, [])
        raise OSError("simulated raw JSONL failure")

    monkeypatch.setattr(acquire, "_write_jsonl", fail_jsonl)
    with pytest.raises(OSError, match="simulated raw JSONL"):
        acquire.publish_acquisition(root, _collection(), _records(), [])
    assert not list(root.rglob("collection_251.json"))
    assert not list(root.rglob("*.tmp"))


def test_output_path_containment_is_enforced(tmp_path: Path) -> None:
    root = tmp_path / "hiba"
    with pytest.raises(acquire.AcquisitionError, match="escapes HIBA root"):
        acquire.ensure_under_root(tmp_path / "outside.json", root)
    with pytest.raises(inventory.InventoryError, match="escapes HIBA root"):
        inventory._ensure_under_root(tmp_path / "outside.jsonl", root)


def test_inventory_preserves_exact_vocabularies_and_missing_ids() -> None:
    records = _records()
    records[0] = _record(0)
    clinical = records[0]["metadata"]["clinical"]  # type: ignore[index]
    del clinical["patient_id"]  # type: ignore[index]
    clinical["lesion_id"] = ""  # type: ignore[index]
    clinical["diagnosis"] = "Melanoma-like text; do not infer"  # type: ignore[index]
    rows, report = inventory.build_inventory(records)
    assert rows[0]["patient_id"] == ""
    assert rows[0]["patient_id_source_path"] == ""
    assert rows[0]["lesion_id"] == ""
    assert report["patient_id"]["missing"] == 1
    assert report["lesion_id"]["missing"] == 1
    assert report["exact_vocabularies"]["diagnosis"][
        "Melanoma-like text; do not infer"
    ] == 1
    assert report["diagnosis_substring_inference_performed"] is False
    assert report["label_mapping_performed"] is False
    assert report["status"] == "metadata_inventory_pending_human_review"
    assert report["evaluation_approval_assigned"] is False


def test_inventory_csv_has_source_paths_and_no_label_mapping_columns(
    tmp_path: Path,
) -> None:
    rows, report = inventory.build_inventory(_records())
    _, csv_path = inventory.publish_inventory(tmp_path / "hiba", rows, report)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert "diagnosis_source_path" in reader.fieldnames
        assert "modality_source_path" in reader.fieldnames
        assert all("mapped" not in column for column in reader.fieldnames)
        first = next(reader)
    assert first["diagnosis"] == "Exact Diagnosis"
    assert first["diagnosis_source_path"] == "metadata.clinical.diagnosis"


def test_inventory_rejects_duplicate_ids_and_wrong_record_count() -> None:
    with pytest.raises(inventory.InventoryError, match="exactly 1616"):
        inventory.build_inventory([_record(1)])
    records = _records()
    records[-1] = dict(records[0])
    with pytest.raises(inventory.InventoryError, match="Duplicate image ID"):
        inventory.build_inventory(records)


def test_registry_and_protocol_keep_official_release_and_scope_locks() -> None:
    registry_text = Path("configs/dataset_registry.yaml").read_text(
        encoding="utf-8"
    )
    protocol = Path(
        "reports/phase10/hiba_official_acquisition_protocol.md"
    ).read_text(encoding="utf-8")
    assert "ISIC collection 251" in registry_text
    assert "10.34970/587329" in registry_text
    assert "expected_image_count: 1616" in registry_text
    assert "Collection `175`" in protocol
    assert "image download" in protocol
    assert "metadata_inventory_pending_human_review" in protocol
    assert (
        "No HIBA metadata or later result may influence model development"
        in " ".join(protocol.split())
    )

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from scripts.build_emb_stage03_split import build_split
from scripts.acquire_isic_stage03_vm import (
    download_one,
    fetch_json,
    inventory_row,
    load_jsonl,
    run_download,
)
from src.data.emb_stage03 import (
    EMBStage03Dataset,
    EMB_STAGE03_CLASS_TO_INDEX,
    derive_t_category_from_isic_metadata,
    inverse_frequency_class_weights,
    map_stage_ajcc,
)
from src.models.efficientnet_baseline import build_efficientnet_b0
from src.training.baseline_experiment import load_experiment_config


def fixture_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage in range(5):
        for index in range(20):
            rows.append(
                {
                    "image_id": f"stage{stage}_{index}",
                    "image_path": f"data/raw/emb/images/stage{stage}_{index}.jpg",
                    "derived_stage_ajcc": stage,
                    "t_category": map_stage_ajcc(stage),
                    "modality": "dermoscopic",
                    "file_sha256": f"{stage:02x}{index:062x}",
                    "eligible": "true",
                    "patient_id": "",
                    "lesion_id": "",
                }
            )
        rows.append(
            {
                "image_id": f"clinical_{stage}",
                "image_path": f"data/raw/emb/images/clinical_{stage}.jpg",
                "derived_stage_ajcc": stage,
                "t_category": map_stage_ajcc(stage),
                "modality": "clinical",
                "file_sha256": f"ff{stage:062x}",
                "eligible": "true",
                "patient_id": "",
                "lesion_id": "",
            }
        )
    return pd.DataFrame(rows)


def test_official_stage_mapping_and_rejection() -> None:
    assert [map_stage_ajcc(value) for value in range(5)] == ["Tis", "T1", "T2", "T3", "T4"]
    for invalid in (None, "", -1, 5, 1.5, "unknown"):
        with pytest.raises(ValueError):
            map_stage_ajcc(invalid)


@pytest.mark.parametrize(
    ("diagnosis", "thickness", "expected"),
    [
        ("Melanoma in situ", "", (0, "Tis")),
        ("Melanoma invasive", 1.0, (1, "T1")),
        ("Melanoma invasive", 1.0001, (2, "T2")),
        ("Melanoma invasive", 2.0, (2, "T2")),
        ("Melanoma invasive", 2.0001, (3, "T3")),
        ("Melanoma invasive", 4.0, (3, "T3")),
        ("Melanoma invasive", 4.0001, (4, "T4")),
    ],
)
def test_official_isic_t_category_boundaries(
    diagnosis: str, thickness: object, expected: tuple[int, str]
) -> None:
    assert derive_t_category_from_isic_metadata(diagnosis, thickness) == expected


@pytest.mark.parametrize(
    ("diagnosis", "thickness"),
    [
        ("Melanoma invasive", ""),
        ("Melanoma invasive", 0),
        ("Melanoma invasive", "unknown"),
        ("Melanoma NOS", 1.0),
        ("Nevus", 1.0),
    ],
)
def test_official_isic_invalid_label_metadata_rejected(
    diagnosis: str, thickness: object
) -> None:
    with pytest.raises(ValueError):
        derive_t_category_from_isic_metadata(diagnosis, thickness)


def api_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "isic_id": "ISIC_0000001",
        "public": True,
        "copyright_license": "CC-BY",
        "attribution": "Anonymous",
        "files": {"full": {"url": "https://example.invalid/image.jpg", "size": 123}},
        "metadata": {
            "acquisition": {"image_type": "dermoscopic"},
            "clinical": {
                "diagnosis_2": "Malignant",
                "diagnosis_3": "Melanoma invasive",
                "diagnosis_confirm_type": "histopathology",
                "mel_thick_mm": 1.2,
                "mel_ulcer": False,
                "patient_id": "patient-1",
                "lesion_id": "lesion-1",
            },
        },
    }
    payload.update(overrides)
    return payload


def test_inventory_eligibility_and_audit_only_disagreement() -> None:
    candidate = {"image": "ISIC_0000001", "stage_ajcc": "1"}
    row = inventory_row(candidate, api_payload())
    assert row["eligible"] == "true"
    assert row["attribution"] == "Anonymous"
    assert row["patient_id"] == "patient-1"
    assert row["lesion_id"] == "lesion-1"
    assert row["derived_stage_ajcc"] == 2
    assert row["t_category"] == "T2"
    assert row["original_emb_t_category"] == "T1"
    assert row["original_vs_official_agreement"] == "false"


def test_inventory_excludes_non_public_modality_and_licence() -> None:
    candidate = {"image": "ISIC_0000001", "stage_ajcc": "2"}
    non_public = api_payload(public=False)
    row = inventory_row(candidate, non_public)
    assert row["eligible"] == "false"
    assert "not_public" in row["exclusion_reason"]

    clinical = api_payload()
    clinical["metadata"]["acquisition"]["image_type"] = "clinical"  # type: ignore[index]
    row = inventory_row(candidate, clinical)
    assert "not_dermoscopic" in row["exclusion_reason"]

    unsupported = api_payload(copyright_license="All rights reserved")
    row = inventory_row(candidate, unsupported)
    assert "unsupported_or_missing_licence" in row["exclusion_reason"]


def test_metadata_resume_loads_existing_jsonl_without_network(tmp_path: Path) -> None:
    path = tmp_path / "resume.jsonl"
    record = {"requested_isic_id": "ISIC_0000001", "payload": api_payload()}
    path.write_text(__import__("json").dumps(record) + "\n", encoding="utf-8")
    assert load_jsonl(path) == {"ISIC_0000001": record}


def test_api_fetch_uses_mocked_response_only() -> None:
    class MockResponse:
        def __enter__(self) -> "MockResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return __import__("json").dumps(api_payload()).encode()

    calls: list[str] = []

    def opener(request: object, timeout: float) -> MockResponse:
        calls.append(f"{getattr(request, 'full_url')}:{timeout}")
        return MockResponse()

    payload = fetch_json("ISIC_0000001", 5.0, 0, opener=opener)
    assert payload["isic_id"] == "ISIC_0000001"
    assert calls == [
        "https://api.isic-archive.com/api/v2/images/ISIC_0000001/:5.0"
    ]


def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


class DownloadResponse(io.BytesIO):
    def __enter__(self) -> "DownloadResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


@pytest.mark.parametrize("size_delta", [0, 7])
def test_download_one_accepts_verified_image_size_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size_delta: int
) -> None:
    content = jpeg_bytes()
    row = {
        "image_id": "ISIC_TEST",
        "full_image_url": "https://example.invalid/image.jpg",
        "expected_file_size": str(len(content) + size_delta),
    }
    monkeypatch.setattr(
        "scripts.acquire_isic_stage03_vm.urllib.request.urlopen",
        lambda request, timeout: DownloadResponse(content),
    )

    path, digest, status = download_one(row, tmp_path, timeout=1.0, retries=0)

    assert Path(path).read_bytes() == content
    assert digest == hashlib.sha256(content).hexdigest()
    assert status == ("downloaded" if size_delta == 0 else "downloaded_size_mismatch")
    assert not (tmp_path / "ISIC_TEST.jpg.part").exists()


def test_download_one_rejects_corrupt_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"not an image"
    row = {
        "image_id": "ISIC_CORRUPT",
        "full_image_url": "https://example.invalid/image.jpg",
        "expected_file_size": str(len(content) + 1),
    }
    monkeypatch.setattr(
        "scripts.acquire_isic_stage03_vm.urllib.request.urlopen",
        lambda request, timeout: DownloadResponse(content),
    )

    with pytest.raises(OSError):
        download_one(row, tmp_path, timeout=1.0, retries=0)

    assert not (tmp_path / "ISIC_CORRUPT.jpg").exists()
    assert not (tmp_path / "ISIC_CORRUPT.jpg.part").exists()


def test_download_one_reports_existing_valid_size_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = jpeg_bytes()
    final = tmp_path / "ISIC_EXISTING.jpg"
    final.write_bytes(content)
    row = {
        "image_id": "ISIC_EXISTING",
        "full_image_url": "https://example.invalid/image.jpg",
        "expected_file_size": str(len(content) - 1),
    }

    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("valid existing image attempted a network call")

    monkeypatch.setattr(
        "scripts.acquire_isic_stage03_vm.urllib.request.urlopen", fail_network
    )

    path, digest, status = download_one(row, tmp_path, timeout=1.0, retries=0)

    assert Path(path) == final
    assert digest == hashlib.sha256(content).hexdigest()
    assert status == "existing_valid_size_mismatch"


@pytest.mark.parametrize(
    "status", ["downloaded_size_mismatch", "existing_valid_size_mismatch"]
)
def test_download_resume_accepts_size_mismatch_status_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    image_dir = tmp_path / "data/raw/emb/images/isic"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "ISIC_RESUME.jpg"
    image_path.write_bytes(jpeg_bytes())
    inventory = tmp_path / "data/raw/emb/isic_stage03_official_inventory.csv"
    pd.DataFrame(
        [
            {
                "image_id": "ISIC_RESUME",
                "full_image_url": "https://example.invalid/image.jpg",
                "expected_file_size": "1",
                "eligible": "true",
                "image_path": image_path.relative_to(tmp_path).as_posix(),
                "file_sha256": "existing",
                "download_status": status,
            }
        ]
    ).to_csv(inventory, index=False)

    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("resume attempted a network call")

    monkeypatch.setattr(
        "scripts.acquire_isic_stage03_vm.urllib.request.urlopen", fail_network
    )
    args = argparse.Namespace(
        project_root=tmp_path, resume=True, workers=1, timeout=1.0, retries=0
    )

    assert run_download(args) == 0
    result = pd.read_csv(inventory, dtype=str, keep_default_na=False).iloc[0]
    assert result["download_status"] == status


def test_split_is_deterministic_stratified_dermoscopic_and_disjoint() -> None:
    first, weights, limitations = build_split(fixture_frame(), seed=42)
    second, second_weights, _ = build_split(fixture_frame(), seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert weights == second_weights
    assert set(first["modality"]) == {"dermoscopic"}
    assert set(first["split"]) == {"train", "validation", "test"}
    assert first.groupby("split")["image_id"].nunique().sum() == first["image_id"].nunique()
    assert first.groupby("file_sha256")["split"].nunique().max() == 1
    assert set(first.groupby("split")["t_category"].nunique()) == {5}
    assert limitations


def test_duplicate_hash_stays_in_one_split() -> None:
    frame = fixture_frame()
    duplicate = frame.iloc[[0]].copy()
    duplicate["image_id"] = "duplicate_copy"
    frame = pd.concat([frame, duplicate], ignore_index=True)
    manifest, _, _ = build_split(frame)
    assert manifest.loc[manifest["file_sha256"] == frame.iloc[0]["file_sha256"], "split"].nunique() == 1


def test_train_only_class_weights() -> None:
    labels = ["Tis"] * 10 + ["T1"] * 5 + ["T2"] * 4 + ["T3"] * 2 + ["T4"]
    weights = inverse_frequency_class_weights(labels)
    assert list(weights) == list(EMB_STAGE03_CLASS_TO_INDEX)
    assert weights["T4"] > weights["Tis"]
    with pytest.raises(ValueError):
        inverse_frequency_class_weights(["Tis", "T1", "T2", "T3"])


def test_five_class_output_shape() -> None:
    model = build_efficientnet_b0(5, pretrained="none")
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros(2, 3, 224, 224))
    assert logits.shape == (2, 5)


def test_phase09_config_resolves() -> None:
    config = load_experiment_config(
        Path("configs/experiments/phase09_stage03_emb_efficientnet_b0_cross_entropy.yaml")
    )
    assert config["data"]["class_to_index"] == {"Tis": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 4}
    assert config["data"]["modality"] == "dermoscopic"
    assert config["data"]["dataset"] == "isic_stage03"
    assert config["data"]["label_source"] == "official_isic_metadata"
    assert config["training"]["epochs"] == 30


def test_loader_accepts_isic_stage03_manifest(tmp_path: Path) -> None:
    rows = []
    for split in ("train", "validation", "test"):
        for stage in range(5):
            rows.append(
                {
                    "dataset": "isic_stage03",
                    "image_id": f"{split}_{stage}",
                    "image_path": f"unused/{split}_{stage}.jpg",
                    "derived_stage_ajcc": stage,
                    "t_category": map_stage_ajcc(stage),
                    "modality": "dermoscopic",
                    "split": split,
                    "split_group_id": f"group_{split}_{stage}",
                    "file_sha256": f"{stage}{split}",
                }
            )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    dataset = EMBStage03Dataset(manifest, tmp_path, "train")
    assert len(dataset) == 5
    assert dataset.class_counts() == {"Tis": 1, "T1": 1, "T2": 1, "T3": 1, "T4": 1}

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from src.data.dataloaders import DataLoaderConfig
from src.data.hierarchical_dataloader import (
    build_hierarchical_inference_dataloader,
)
from src.evaluation.hierarchical_protocol import (
    FrozenCheckpointSpec,
    load_hierarchical_evaluation_protocol,
    verify_frozen_checkpoint,
)


PROJECT_CONFIG = Path(
    "configs/evaluation/phase05_hierarchical_internal_test.yaml"
)


def _load_repository_config() -> dict[str, object]:
    loaded = yaml.safe_load(
        PROJECT_CONFIG.read_text(encoding="utf-8-sig")
    )
    assert isinstance(loaded, dict)
    return loaded


def _write_yaml(
    path: Path,
    payload: dict[str, object],
) -> Path:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_repository_phase05_protocol_is_locked() -> None:
    protocol = load_hierarchical_evaluation_protocol(
        PROJECT_CONFIG,
        project_root=Path.cwd(),
    )

    assert protocol.seed == 42
    assert protocol.stage_1.epoch == 5
    assert protocol.stage_2.epoch == 8
    assert protocol.stage_1.class_names == (
        "non_malignant",
        "malignant",
    )
    assert protocol.stage_2.class_names == (
        "melanoma",
        "bcc",
        "scc",
    )
    assert protocol.final_class_to_index == {
        "non_malignant": 0,
        "melanoma": 1,
        "bcc": 2,
        "scc": 3,
    }
    assert protocol.output_directory.name == (
        "locked_primary_evaluation"
    )


def test_phase05_protocol_rejects_changed_gate_policy(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(_load_repository_config())
    payload["hierarchy"]["gate_policy"] = "threshold"

    path = _write_yaml(
        tmp_path / "phase05.yaml",
        payload,
    )

    with pytest.raises(
        ValueError,
        match="gate_policy",
    ):
        load_hierarchical_evaluation_protocol(
            path,
            project_root=tmp_path,
        )


def _write_checkpoint_fixture(
    tmp_path: Path,
) -> tuple[Path, str]:
    checkpoint_directory = tmp_path / "run"
    checkpoint_directory.mkdir()

    checkpoint_path = checkpoint_directory / "best_checkpoint.pt"
    payload = {
        "epoch": 5,
        "model_state_dict": {},
        "validation_metrics": {
            "macro_f1": 0.8,
        },
        "class_names": [
            "non_malignant",
            "malignant",
        ],
        "config": {
            "runtime": {
                "sanity_run": False,
            },
            "data": {
                "task": "stage_1",
                "class_to_index": {
                    "non_malignant": 0,
                    "malignant": 1,
                },
            },
            "model": {
                "architecture": "efficientnet_b0",
                "number_of_classes": 2,
                "dropout_probability": 0.2,
            },
        },
    }
    torch.save(payload, checkpoint_path)

    (checkpoint_directory / "run_summary.json").write_text(
        json.dumps(
            {
                "reportable_as_full_result": True,
                "sanity_run": False,
                "best_epoch": 5,
                "task": "stage_1",
            }
        ),
        encoding="utf-8",
    )

    digest = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()

    return checkpoint_path, digest


def test_frozen_checkpoint_identity_validation_accepts_full_run(
    tmp_path: Path,
) -> None:
    checkpoint_path, digest = _write_checkpoint_fixture(
        tmp_path
    )

    specification = FrozenCheckpointSpec(
        task="stage_1",
        path=checkpoint_path,
        sha256=digest,
        epoch=5,
        class_to_index={
            "non_malignant": 0,
            "malignant": 1,
        },
        class_names=(
            "non_malignant",
            "malignant",
        ),
    )

    loaded = verify_frozen_checkpoint(specification)

    assert loaded["epoch"] == 5
    assert loaded["class_names"] == [
        "non_malignant",
        "malignant",
    ]


def test_frozen_checkpoint_identity_validation_rejects_hash_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint_path, _ = _write_checkpoint_fixture(
        tmp_path
    )

    specification = FrozenCheckpointSpec(
        task="stage_1",
        path=checkpoint_path,
        sha256="0" * 64,
        epoch=5,
        class_to_index={
            "non_malignant": 0,
            "malignant": 1,
        },
        class_names=(
            "non_malignant",
            "malignant",
        ),
    )

    with pytest.raises(
        ValueError,
        match="SHA-256 mismatch",
    ):
        verify_frozen_checkpoint(specification)


def _manifest_row(
    image_id: str,
    image_path: str,
    stage_1_label: str,
    stage_2_label: str,
) -> dict[str, str]:
    malignant = stage_1_label == "malignant"

    return {
        "dataset": "isic2019",
        "image_id": image_id,
        "image_path": image_path,
        "split": "internal_test",
        "split_included": "1",
        "split_group_id": f"group_{image_id}",
        "include_stage_1": "1",
        "include_stage_2": "1" if malignant else "0",
        "stage_1_label": stage_1_label,
        "stage_2_label": stage_2_label,
        "file_sha256": f"hash_{image_id}",
    }


def test_hierarchical_dataloader_preserves_manifest_order(
    tmp_path: Path,
) -> None:
    cases = [
        ("first", "non_malignant", ""),
        ("second", "malignant", "melanoma"),
        ("third", "malignant", "scc"),
    ]

    rows: list[dict[str, str]] = []

    for image_id, stage_1_label, stage_2_label in cases:
        relative_path = Path("images") / f"{image_id}.jpg"
        image_path = tmp_path / relative_path
        image_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        Image.new(
            "RGB",
            (16, 16),
            color=(100, 120, 140),
        ).save(image_path)

        rows.append(
            _manifest_row(
                image_id,
                str(relative_path),
                stage_1_label,
                stage_2_label,
            )
        )

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(
        manifest_path,
        index=False,
    )

    loader = build_hierarchical_inference_dataloader(
        manifest_path,
        tmp_path,
        config=DataLoaderConfig(
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            seed=42,
        ),
        verify_image_paths=True,
    )

    observed_ids: list[str] = []
    observed_targets: list[int] = []

    for batch in loader:
        observed_ids.extend(batch["image_id"])
        observed_targets.extend(
            batch["final_target"].tolist()
        )

    assert observed_ids == [
        "first",
        "second",
        "third",
    ]
    assert observed_targets == [0, 1, 3]

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.data.flat_four_class_audit import audit_flat_four_class_manifest
from src.training.baseline_experiment import (
    compute_effective_number_class_weights,
    load_experiment_config,
)


FOCAL_CONFIG = Path(
    "configs/experiments/"
    "phase06b_flat_four_class_isic2019_efficientnet_b0_"
    "class_balanced_focal_loss.yaml"
)
CLEAN_CONFIG = Path(
    "configs/experiments/"
    "phase06_flat_four_class_isic2019_efficientnet_b0_cross_entropy.yaml"
)
CLASS_ORDER = ["non_malignant", "melanoma", "bcc", "scc"]
TRAIN_COUNTS = {
    "non_malignant": 11193,
    "melanoma": 3164,
    "bcc": 2327,
    "scc": 440,
}
EXPECTED_WEIGHTS = {
    "non_malignant": 0.1787906368601743,
    "melanoma": 0.4439450826095749,
    "bcc": 0.579994522856464,
    "scc": 2.797269757673787,
}


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_locked_manifest_training_counts_and_order() -> None:
    audit = audit_flat_four_class_manifest(
        "data/manifests/isic2019_train_val_test_split_seed42.csv"
    )
    observed = {
        name: audit["counts"]["train"]["classes"][name]["count"]
        for name in audit["class_order"]
    }

    assert audit["class_order"] == CLASS_ORDER
    assert observed == TRAIN_COUNTS
    assert sum(observed.values()) == 17124


def test_phase06b_focal_metadata_and_weights_are_aligned_and_deterministic() -> None:
    config = load_experiment_config(FOCAL_CONFIG)
    training = config["training"]
    source = training["class_weight_source"]

    first = compute_effective_number_class_weights(
        source["class_counts"], CLASS_ORDER, beta=source["beta"]
    )
    second = compute_effective_number_class_weights(
        source["class_counts"], CLASS_ORDER, beta=source["beta"]
    )

    assert config["data"]["class_to_index"] == dict(zip(CLASS_ORDER, range(4)))
    assert config["model"]["number_of_classes"] == 4
    assert training["loss"] == "class_balanced_focal_loss"
    assert training["focal_loss"] is True
    assert source["beta"] == 0.9999
    assert training["focal_gamma"] == 2.0
    assert source["class_counts"] == TRAIN_COUNTS
    assert list(training["class_weights"]) == CLASS_ORDER
    assert first == second
    assert first == pytest.approx(EXPECTED_WEIGHTS, rel=1e-12, abs=1e-12)
    assert training["class_weights"] == pytest.approx(first, rel=1e-12, abs=1e-12)
    assert all(weight > 0 for weight in first.values())
    assert first["scc"] == max(first.values())


def test_phase06b_is_a_loss_only_change_from_clean_ce() -> None:
    clean = yaml.safe_load(CLEAN_CONFIG.read_text(encoding="utf-8"))
    focal = yaml.safe_load(FOCAL_CONFIG.read_text(encoding="utf-8"))

    clean_identity = deepcopy(clean)
    focal_identity = deepcopy(focal)
    for payload in (clean_identity, focal_identity):
        payload["experiment"].pop("research_stage")
        payload["experiment"].pop("run_name")
        payload["experiment"].pop("variant")
        payload.pop("training")

    assert focal_identity == clean_identity

    clean_training = deepcopy(clean["training"])
    focal_training = deepcopy(focal["training"])
    for key in (
        "loss",
        "class_weights",
        "class_weight_source",
        "focal_loss",
        "focal_gamma",
    ):
        clean_training.pop(key, None)
        focal_training.pop(key, None)
    assert focal_training == clean_training
    assert load_experiment_config(CLEAN_CONFIG)["training"]["loss"] == "cross_entropy"


@pytest.mark.parametrize(
    "path,task,class_order",
    [
        (
            "configs/experiments/"
            "phase03_stage01_isic2019_efficientnet_b0_cross_entropy.yaml",
            "stage_1",
            ["non_malignant", "malignant"],
        ),
        (
            "configs/experiments/"
            "phase03_stage02_isic2019_efficientnet_b0_cross_entropy.yaml",
            "stage_2",
            ["melanoma", "bcc", "scc"],
        ),
        (
            "configs/experiments/"
            "phase04_stage02_isic2019_efficientnet_b0_class_balanced_focal_loss.yaml",
            "stage_2",
            ["melanoma", "bcc", "scc"],
        ),
    ],
)
def test_existing_task_configs_remain_unchanged(
    path: str, task: str, class_order: list[str]
) -> None:
    config = load_experiment_config(path)

    assert config["data"]["task"] == task
    assert list(config["data"]["class_to_index"]) == class_order


def test_existing_three_class_focal_weights_remain_numerically_unchanged() -> None:
    config = load_experiment_config(
        "configs/experiments/"
        "phase04_stage02_isic2019_efficientnet_b0_class_balanced_focal_loss.yaml"
    )
    training = config["training"]

    assert training["class_weights"] == pytest.approx(
        {
            "melanoma": 0.3485376280807543,
            "bcc": 0.4553489231324597,
            "scc": 2.196113448786786,
        },
        rel=1e-12,
        abs=1e-12,
    )
    assert training["class_weight_source"]["beta"] == 0.9999
    assert training["focal_gamma"] == 2.0


def test_invalid_count_alignment_fails_clearly() -> None:
    with pytest.raises(ValueError, match="same classes"):
        compute_effective_number_class_weights(
            {"non_malignant": 1, "melanoma": 1},
            CLASS_ORDER,
            beta=0.9999,
        )


def test_non_positive_counts_fail_clearly() -> None:
    counts = dict(TRAIN_COUNTS)
    counts["scc"] = 0
    with pytest.raises(ValueError, match="positive"):
        compute_effective_number_class_weights(counts, CLASS_ORDER, beta=0.9999)


def test_config_persists_focal_provenance_and_rejects_wrong_weight(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(FOCAL_CONFIG.read_text(encoding="utf-8"))
    payload["training"]["class_weights"]["scc"] = 1.0

    with pytest.raises(ValueError, match="effective-number formula"):
        load_experiment_config(_write_config(tmp_path, payload))

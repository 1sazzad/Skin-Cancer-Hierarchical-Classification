from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import pytest
import yaml


CONFIG = Path("configs/evaluation/phase06c_selected_flat_internal_test.yaml")
REPORT = Path("reports/phase06/phase06c_selected_flat_internal_test_protocol.md")
RESULT_REPORT = Path("reports/phase06/phase06b_class_balanced_focal_amendment.md")
REGISTRY = Path("experiments/experiment_registry.csv")
SELECTED_PATH = (
    "runs/phase06_full/"
    "full__phase06_flat_four_class_isic2019_efficientnet_b0_cross_entropy_"
    "seed42__20260726T232308Z/best_checkpoint.pt"
)
SELECTED_HASH = "f3d8b8b0e5ef42e3c287a2377b5570411d442246acd16cb874ccf903facdc7a7"
REJECTED_PATH = (
    "runs/phase06b/full/"
    "full__phase06b_flat_four_class_isic2019_efficientnet_b0_"
    "class_balanced_focal_loss_seed42__20260727T120615Z/best_checkpoint.pt"
)
REJECTED_HASH = "07586d515cd9378e05831ca542f391e32b3b7a6c669c7dd83ce1df219b2af015"


def _config() -> dict[str, object]:
    assert CONFIG.is_file()
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _registry_rows() -> dict[str, dict[str, str]]:
    with REGISTRY.open(encoding="utf-8-sig", newline="") as handle:
        return {row["run_id"]: row for row in csv.DictReader(handle)}


def test_protocol_is_prepared_unconsumed_and_single_candidate() -> None:
    payload = _config()

    assert payload["phase"] == "06C"
    assert payload["protocol_status"] == "prepared_not_executed"
    assert payload["internal_test_accessed"] is False
    assert payload["valid_internal_test_run_completed"] is False
    assert payload["allowed_checkpoint_count"] == 1
    assert payload["allowed_candidate_count"] == 1
    assert payload["seed"] == 42
    assert payload["selection_basis"] == "validation_only"
    assert payload["internal_test_split"] == "test"
    assert payload["evaluator_loader_key"] == "internal_test"


def test_selected_and_rejected_checkpoint_identities_are_frozen() -> None:
    payload = _config()
    selected = payload["selected_model"]
    rejected = payload["rejected_candidates"]

    assert selected == {
        "phase": "06A",
        "model": "efficientnet_b0",
        "loss": "cross_entropy",
        "checkpoint_path": SELECTED_PATH,
        "checkpoint_sha256": SELECTED_HASH,
    }
    assert isinstance(rejected, list) and len(rejected) == 1
    assert rejected[0]["phase"] == "06B"
    assert rejected[0]["checkpoint_path"] == REJECTED_PATH
    assert rejected[0]["checkpoint_sha256"] == REJECTED_HASH
    assert rejected[0]["candidate_status"] == "rejected_by_validation_selection"
    assert rejected[0]["internal_test_allowed"] is False
    assert selected["checkpoint_sha256"] != REJECTED_HASH


def test_tuning_switching_and_local_execution_are_forbidden() -> None:
    payload = _config()

    assert payload["candidate_switching_allowed"] is False
    assert payload["post_test_tuning_allowed"] is False
    assert payload["threshold_tuning_allowed"] is False
    assert payload["local_model_execution_allowed"] is False
    assert payload["required_execution_environment"] == "azure_tesla_t4"


def test_preexecution_protocol_contains_no_internal_test_metrics() -> None:
    payload = _config()
    forbidden = {"accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                 "precision", "recall", "f1", "confusion_matrix", "predictions"}

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert not (keys(payload) & forbidden)


def test_executable_evaluation_command_uses_only_selected_checkpoint() -> None:
    text = REPORT.read_text(encoding="utf-8")
    bash_blocks = re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL)
    evaluation_blocks = [block for block in bash_blocks if "evaluate_isic2019" in block]

    assert len(evaluation_blocks) == 1
    command = evaluation_blocks[0]
    assert SELECTED_PATH in command
    assert REJECTED_PATH not in command
    assert REJECTED_HASH not in command
    assert "--checkpoint '$CHECKPOINT'" in command
    assert "*" not in next(
        line for line in command.splitlines() if "evaluate_isic2019" in line
    )


def test_registry_freezes_selection_without_internal_test_access() -> None:
    rows = _registry_rows()
    clean = rows["phase06_flat_four_class_clean_ce_seed42"]
    focal = rows["phase06b_flat_four_class_cb_focal_seed42"]

    assert clean["status"] == focal["status"] == "completed_validation"
    assert clean["checkpoint_path"] == SELECTED_PATH
    assert clean["primary_metric_value"] == "0.6535716654"
    assert "Selected for Phase 06C=true" in clean["notes"]
    assert "internal test accessed=false" in clean["notes"]
    assert focal["checkpoint_path"] == REJECTED_PATH
    assert focal["primary_metric_value"] == "0.6490067298"
    assert "candidate rejected by validation selection" in focal["notes"]
    assert "selected for Phase 06C=false" in focal["notes"]
    assert "internal test accessed=false" in focal["notes"]


def test_result_report_records_validation_only_selection() -> None:
    text = RESULT_REPORT.read_text(encoding="utf-8")
    assert "Validation-only model selection" in text
    assert "internal test remained untouched" in text
    assert "Only the selected Phase 06A clean-CE checkpoint is eligible" in text


def test_selected_checkpoint_raw_sha256_when_present() -> None:
    checkpoint = Path(SELECTED_PATH)
    if not checkpoint.is_file():
        pytest.skip("Selected checkpoint is artifact-managed and absent locally.")

    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == SELECTED_HASH

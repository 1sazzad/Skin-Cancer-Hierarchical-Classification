from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import pytest
import yaml


CONFIG = Path("configs/evaluation/phase06c_selected_flat_internal_test.yaml")
PROTOCOL_REPORT = Path(
    "reports/phase06/phase06c_selected_flat_internal_test_protocol.md"
)
SELECTION_REPORT = Path(
    "reports/phase06/phase06b_class_balanced_focal_amendment.md"
)
RESULT_REPORT = Path(
    "reports/phase06/phase06c_selected_flat_internal_test_result.md"
)
REGISTRY = Path("experiments/experiment_registry.csv")

SELECTED_PATH = "runs/phase06_full/full__phase06_flat_four_class_isic2019_efficientnet_b0_cross_entropy_seed42__20260726T232308Z/best_checkpoint.pt"
SELECTED_HASH = "f3d8b8b0e5ef42e3c287a2377b5570411d442246acd16cb874ccf903facdc7a7"
REJECTED_PATH = (
    "runs/phase06b/full/"
    "full__phase06b_flat_four_class_isic2019_efficientnet_b0_"
    "class_balanced_focal_loss_seed42__20260727T120615Z/best_checkpoint.pt"
)
REJECTED_HASH = (
    "07586d515cd9378e05831ca542f391e32b3b7a6c669c7dd83ce1df219b2af015"
)

EVALUATION_COMMIT = "550e7cdb1144f059c940d4240fe4579e0280a803"
ARCHIVE_PATH = "runs/backups/phase06c/phase06c_selected_flat_internal_test_550e7cdb1144.tar.gz"
ARCHIVE_HASH = "b76762b53a35a8d9b0aa96621d78ea0e4421aa6e8052d068ffc10648a4e63e91"
PHASE06C_RUN_ID = "phase06c_selected_flat_internal_test_seed42"


def _config() -> dict[str, object]:
    assert CONFIG.is_file()
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _registry_rows() -> dict[str, dict[str, str]]:
    with REGISTRY.open(encoding="utf-8-sig", newline="") as handle:
        return {row["run_id"]: row for row in csv.DictReader(handle)}


def test_protocol_is_consumed_locked_and_single_candidate() -> None:
    payload = _config()

    assert payload["phase"] == "06C"
    assert payload["protocol_status"] == "consumed_locked"
    assert payload["internal_test_accessed"] is True
    assert payload["valid_internal_test_run_completed"] is True
    assert payload["protocol_consumed"] is True
    assert payload["rerun_allowed"] is False
    assert payload["local_backup_verified"] is True
    assert payload["evaluation_attempt_count"] == 1
    assert payload["valid_internal_test_run_count"] == 1
    assert payload["allowed_checkpoint_count"] == 1
    assert payload["allowed_candidate_count"] == 1
    assert payload["seed"] == 42
    assert payload["selection_basis"] == "validation_only"
    assert payload["internal_test_split"] == "test"
    assert payload["evaluator_loader_key"] == "internal_test"


def test_selected_and_rejected_checkpoint_identities_remain_frozen() -> None:
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


def test_tuning_switching_local_execution_and_rerun_are_forbidden() -> None:
    payload = _config()

    assert payload["candidate_switching_allowed"] is False
    assert payload["post_test_tuning_allowed"] is False
    assert payload["threshold_tuning_allowed"] is False
    assert payload["local_model_execution_allowed"] is False
    assert payload["rerun_allowed"] is False
    assert payload["required_execution_environment"] == "azure_tesla_t4"


def test_consumed_protocol_records_locked_result() -> None:
    payload = _config()
    result = payload["locked_result"]

    assert payload["evaluation_git_commit"] == EVALUATION_COMMIT
    assert result["checkpoint_epoch"] == 2
    assert result["sample_count"] == 3668
    assert result["accuracy"] == pytest.approx(0.7420937840785169)
    assert result["balanced_accuracy"] == pytest.approx(
        0.6503125394090663
    )
    assert result["macro_f1"] == pytest.approx(0.6192224685168973)
    assert result["weighted_f1"] == pytest.approx(0.7525567213826209)
    assert result["mean_loss"] == pytest.approx(0.6232672185518230)
    assert result["metrics_path"] == "runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/internal_test_metrics.json"
    assert result["predictions_path"] == "runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/internal_test_predictions.csv"
    assert result["local_backup_archive"] == ARCHIVE_PATH
    assert result["local_backup_archive_sha256"] == ARCHIVE_HASH
    assert result["artifact_manifest_entry_count"] == 12


def test_historical_evaluation_command_uses_only_selected_checkpoint() -> None:
    text = PROTOCOL_REPORT.read_text(encoding="utf-8")
    bash_blocks = re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL)
    evaluation_blocks = [
        block for block in bash_blocks
        if "evaluate_isic2019" in block
    ]

    assert len(evaluation_blocks) == 1
    command = evaluation_blocks[0]
    assert SELECTED_PATH in command
    assert REJECTED_PATH not in command
    assert REJECTED_HASH not in command
    assert "--checkpoint '$CHECKPOINT'" in command
    assert "*" not in next(
        line for line in command.splitlines()
        if "evaluate_isic2019" in line
    )
    assert "consumed; do not rerun" in text


def test_registry_records_selection_rejection_and_locked_evaluation() -> None:
    rows = _registry_rows()

    clean = rows["phase06_flat_four_class_clean_ce_seed42"]
    focal = rows["phase06b_flat_four_class_cb_focal_seed42"]
    locked = rows[PHASE06C_RUN_ID]

    assert clean["status"] == "completed_validation"
    assert clean["checkpoint_path"] == SELECTED_PATH
    assert clean["primary_metric_value"] == "0.6535716654"
    assert "Selected for Phase 06C=true" in clean["notes"]
    assert "internal test accessed=true" in clean["notes"]
    assert "one-time protocol consumed=true" in clean["notes"]

    assert focal["status"] == "completed_validation"
    assert focal["checkpoint_path"] == REJECTED_PATH
    assert focal["primary_metric_value"] == "0.6490067298"
    assert "candidate rejected by validation selection" in focal["notes"]
    assert "selected for Phase 06C=false" in focal["notes"]
    assert "internal test accessed=false" in focal["notes"]

    assert locked["status"] == "completed_locked"
    assert locked["git_commit"] == EVALUATION_COMMIT[:7]
    assert locked["primary_metric"] == "internal_test_macro_f1"
    assert float(locked["primary_metric_value"]) == pytest.approx(
        0.6192224685168973
    )
    assert locked["checkpoint_path"] == SELECTED_PATH
    assert locked["predictions_path"] == "runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/internal_test_predictions.csv"
    assert locked["metrics_path"] == "runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/internal_test_metrics.json"
    assert ARCHIVE_HASH in locked["notes"]
    assert "rerun allowed=false" in locked["notes"]


def test_validation_only_selection_remains_documented() -> None:
    text = SELECTION_REPORT.read_text(encoding="utf-8")

    assert "Validation-only model selection" in text
    assert "internal test remained untouched" in text
    assert "Only the selected Phase 06A clean-CE checkpoint is eligible" in text


def test_result_report_records_locked_outcome_and_claims_boundary() -> None:
    text = RESULT_REPORT.read_text(encoding="utf-8")

    assert "consumed_locked" in text
    assert "The locked internal-test split contained `3668` images." in text
    assert "Macro-F1 | 0.6192224685" in text
    assert "The Phase 06C protocol has been consumed" in text
    assert ARCHIVE_PATH in text
    assert ARCHIVE_HASH in text
    assert "does not by itself establish statistical significance" in text
    normalized_text = " ".join(text.split())
    assert "rejected Phase 06B focal candidate remains prohibited" in normalized_text


def test_selected_checkpoint_raw_sha256_when_present() -> None:
    checkpoint = Path(SELECTED_PATH)

    if not checkpoint.is_file():
        pytest.skip("Selected checkpoint is artifact-managed and absent locally.")

    digest = hashlib.sha256()

    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    assert digest.hexdigest() == SELECTED_HASH

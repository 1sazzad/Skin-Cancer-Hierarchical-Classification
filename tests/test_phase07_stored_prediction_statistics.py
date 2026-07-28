from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.analysis.stored_prediction_statistics import (
    CLASSES,
    StatisticalAnalysisError,
    bootstrap_table,
    build_analysis,
    calculate_metrics,
    correctness_category,
    exact_mcnemar,
    linear_quantile,
    routing_decomposition,
    sha256_file,
    stratified_bootstrap_indices,
    validate_paired_manifest,
    verify_archive_member,
    write_csv,
    write_json,
)
from src.analysis.statistical_protocol import frozen_linear_quantile


def _paired() -> pd.DataFrame:
    target = [0, 0, 1, 1, 2, 2, 3, 3]
    flat = [0, 1, 1, 2, 2, 0, 3, 2]
    hierarchy = [0, 0, 2, 1, 2, 3, 0, 3]
    return pd.DataFrame(
        {
            "sample_id": [f"id_{i}" for i in range(8)],
            "target_label": [CLASSES[i] for i in target],
            "target_index": target,
            "hierarchical_predicted_label": [CLASSES[i] for i in hierarchy],
            "hierarchical_predicted_index": hierarchy,
            "flat_predicted_label": [CLASSES[i] for i in flat],
            "flat_predicted_index": flat,
        }
    )


def _routing() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": [f"id_{i}" for i in range(8)],
            "final_target_index": [0, 0, 1, 1, 2, 2, 3, 3],
            "stage_1_predicted_index": [0, 1, 0, 1, 1, 1, 0, 1],
            "stage_2_executed": [0, 1, 1, 1, 1, 1, 1, 1],
            "stage_2_predicted_index": [np.nan, 0, 1, 2, 2, 0, 3, 3],
            "predicted_gate_correct": [1, 0, 0, 0, 1, 0, 0, 1],
        }
    )


def test_fixed_confusion_metrics_match_sklearn() -> None:
    target = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    prediction = np.array([0, 1, 1, 2, 2, 0, 3, 2])
    metrics = calculate_metrics(target, prediction)
    precision, recall, f1, support = precision_recall_fscore_support(
        target, prediction, labels=[0, 1, 2, 3], zero_division=0
    )
    np.testing.assert_array_equal(
        metrics.confusion_matrix, confusion_matrix(target, prediction, labels=[0, 1, 2, 3])
    )
    assert metrics.accuracy == pytest.approx(accuracy_score(target, prediction))
    assert metrics.balanced_accuracy == pytest.approx(
        balanced_accuracy_score(target, prediction)
    )
    assert metrics.macro_f1 == pytest.approx(
        f1_score(target, prediction, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    )
    assert metrics.weighted_f1 == pytest.approx(
        f1_score(target, prediction, labels=[0, 1, 2, 3], average="weighted", zero_division=0)
    )
    np.testing.assert_allclose(metrics.precision, precision)
    np.testing.assert_allclose(metrics.recall, recall)
    np.testing.assert_allclose(metrics.f1, f1)
    np.testing.assert_array_equal(metrics.support, support)


def test_fixed_labels_and_zero_division() -> None:
    metrics = calculate_metrics(np.array([0, 0]), np.array([0, 0]))
    np.testing.assert_array_equal(metrics.support, [2, 0, 0, 0])
    np.testing.assert_array_equal(metrics.precision[1:], [0, 0, 0])
    np.testing.assert_array_equal(metrics.recall[1:], [0, 0, 0])
    np.testing.assert_array_equal(metrics.f1[1:], [0, 0, 0])


@pytest.mark.parametrize("bad", [-1, 4])
def test_unsupported_labels_fail(bad: int) -> None:
    with pytest.raises(StatisticalAnalysisError, match="labels"):
        calculate_metrics(np.array([0, 1]), np.array([0, bad]))


def test_stratified_indices_preserve_support_and_are_deterministic() -> None:
    target = _paired()["target_index"].to_numpy()
    first = stratified_bootstrap_indices(target, 10, 42)
    second = stratified_bootstrap_indices(target, 10, 42)
    np.testing.assert_array_equal(first, second)
    for row in first:
        np.testing.assert_array_equal(np.bincount(target[row], minlength=4), [2, 2, 2, 2])


def test_bootstrap_has_paired_direction_ids_and_no_skips() -> None:
    paired = _paired()
    table = bootstrap_table(
        paired["target_index"].to_numpy(),
        paired["flat_predicted_index"].to_numpy(),
        paired["hierarchical_predicted_index"].to_numpy(),
        replicate_count=10000,
        seed=42,
    )
    np.testing.assert_array_equal(table["replicate_id"], np.arange(10000))
    assert np.isfinite(table.drop(columns="replicate_id").to_numpy()).all()
    np.testing.assert_allclose(
        table["difference_macro_f1"],
        table["flat_macro_f1"] - table["hierarchical_macro_f1"],
    )


def test_linear_quantile_requires_float64_and_finite() -> None:
    with pytest.raises(StatisticalAnalysisError, match="float64"):
        linear_quantile(np.array([1, 2], dtype=np.float32), [0.5])
    with pytest.raises(StatisticalAnalysisError, match="NaN"):
        linear_quantile(np.array([1.0, np.nan], dtype=np.float64), [0.5])


@pytest.mark.parametrize(
    "values",
    [
        [-3.0, 1.0, 8.0],
        [-4.0, -1.0, 2.0, 10.0],
        [2.0, 2.0, 2.0, 5.0],
        np.linspace(-2, 7, 10000).tolist(),
    ],
)
def test_linear_quantile_matches_independent_formula(values: list[float]) -> None:
    array = np.asarray(values, dtype=np.float64)
    observed = linear_quantile(array, [0.0, 0.025, 0.975, 1.0])
    expected = np.array(
        [frozen_linear_quantile(values, q) for q in [0.0, 0.025, 0.975, 1.0]]
    )
    np.testing.assert_allclose(observed, expected, rtol=0, atol=4e-15)


def test_exact_mcnemar_known_case() -> None:
    result = exact_mcnemar(
        np.array([1] * 8 + [0] * 5, dtype=bool),
        np.array([0] * 8 + [1] * 5, dtype=bool),
    )
    assert result["flat_correct_hierarchy_wrong"] == 8
    assert result["flat_wrong_hierarchy_correct"] == 5
    assert result["exact_two_sided_p_value"] == pytest.approx(0.5810546875)
    assert result["raw_discordant_pair_odds_ratio"]["numeric_value"] == 1.6


def test_mcnemar_no_discordance_and_odds_edges() -> None:
    none = exact_mcnemar(np.array([True]), np.array([True]))
    assert none["exact_two_sided_p_value"] == 1.0
    assert none["raw_discordant_pair_odds_ratio"]["status"] == "undefined"
    infinity = exact_mcnemar(np.array([True]), np.array([False]))
    assert infinity["raw_discordant_pair_odds_ratio"]["status"] == "positive_infinity"
    zero = exact_mcnemar(np.array([False]), np.array([True]))
    assert zero["raw_discordant_pair_odds_ratio"]["numeric_value"] == 0.0


@pytest.mark.parametrize(
    ("flat", "hierarchy", "expected"),
    [
        (True, True, "both_correct"),
        (True, False, "correct_only_flat"),
        (False, True, "correct_only_hierarchy"),
        (False, False, "both_wrong"),
    ],
)
def test_correctness_categories(flat: bool, hierarchy: bool, expected: str) -> None:
    assert correctness_category(flat, hierarchy) == expected


def test_manifest_validation_duplicate_missing_and_mapping() -> None:
    frame = _paired()
    validate_paired_manifest(
        frame, expected_count=8, expected_support=np.array([2, 2, 2, 2])
    )
    duplicate = frame.copy()
    duplicate.loc[1, "sample_id"] = duplicate.loc[0, "sample_id"]
    with pytest.raises(StatisticalAnalysisError, match="Duplicate"):
        validate_paired_manifest(
            duplicate, expected_count=8, expected_support=np.array([2, 2, 2, 2])
        )
    missing = frame.copy()
    missing.loc[0, "sample_id"] = ""
    with pytest.raises(StatisticalAnalysisError, match="Missing"):
        validate_paired_manifest(
            missing, expected_count=8, expected_support=np.array([2, 2, 2, 2])
        )
    mismatch = frame.copy()
    mismatch.loc[0, "target_label"] = "melanoma"
    with pytest.raises(StatisticalAnalysisError, match="mapping"):
        validate_paired_manifest(
            mismatch, expected_count=8, expected_support=np.array([2, 2, 2, 2])
        )


def _make_archive(path: Path, members: list[tuple[str, bytes, str]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, content, kind in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            if kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "target"
                info.size = 0
                archive.addfile(info)
            else:
                archive.addfile(info, io.BytesIO(content))


def test_safe_archive_member_verification(tmp_path: Path) -> None:
    archive = tmp_path / "safe.tar.gz"
    content = b"a,b\n1,2\n"
    _make_archive(archive, [("canonical.csv", content, "file")])
    assert verify_archive_member(
        archive, sha256_file(archive), "canonical.csv", __import__("hashlib").sha256(content).hexdigest()
    ) == content


@pytest.mark.parametrize(
    "members",
    [
        [("../unsafe", b"x", "file"), ("canonical.csv", b"x", "file")],
        [("/absolute", b"x", "file"), ("canonical.csv", b"x", "file")],
        [("link", b"", "symlink"), ("canonical.csv", b"x", "file")],
        [("canonical.csv", b"x", "file"), ("canonical.csv", b"x", "file")],
    ],
)
def test_unsafe_or_duplicate_archive_members_fail(
    tmp_path: Path, members: list[tuple[str, bytes, str]]
) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    _make_archive(archive, members)
    with pytest.raises(StatisticalAnalysisError):
        verify_archive_member(archive, sha256_file(archive), "canonical.csv", "0" * 64)


def test_archive_and_member_hash_mismatches_fail(tmp_path: Path) -> None:
    archive = tmp_path / "archive.tar.gz"
    _make_archive(archive, [("canonical.csv", b"x", "file")])
    with pytest.raises(StatisticalAnalysisError, match="archive"):
        verify_archive_member(archive, "0" * 64, "canonical.csv", "0" * 64)
    with pytest.raises(StatisticalAnalysisError, match="member"):
        verify_archive_member(archive, sha256_file(archive), "canonical.csv", "0" * 64)


def test_routing_decomposition_and_scc_end_to_end() -> None:
    routing = routing_decomposition(_routing())
    counts = dict(zip(routing["category"], routing["count"]))
    assert counts["true_malignant_routed_non_malignant"] == 2
    assert counts["true_non_malignant_routed_stage2"] == 1
    assert counts["structural_stage2_missing_not_invoked"] == 1
    result = build_analysis(
        _paired(),
        _routing(),
        replicate_count=20,
        seed=42,
        expected_count=8,
        expected_support=np.array([2, 2, 2, 2]),
    )
    assert len(result["replicates"]) == 20
    assert set(result["sample_categories"]["correctness_category"]) <= {
        "both_correct",
        "both_wrong",
        "correct_only_flat",
        "correct_only_hierarchy",
    }
    assert int(result["scc"].iloc[0]["support"]) == 2


def test_non_finite_json_and_deterministic_serialization(tmp_path: Path) -> None:
    with pytest.raises(StatisticalAnalysisError, match="NaN"):
        write_json(tmp_path / "bad.json", {"value": float("nan")})
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_json(first, {"b": 2.0, "a": 1})
    write_json(second, {"a": 1, "b": 2.0})
    assert first.read_bytes() == second.read_bytes()
    frame = pd.DataFrame({"value": [1.2345678901234567]})
    write_csv(tmp_path / "first.csv", frame)
    write_csv(tmp_path / "second.csv", frame)
    assert (tmp_path / "first.csv").read_bytes() == (tmp_path / "second.csv").read_bytes()


def test_changed_frozen_configuration_is_rejected_by_hash(tmp_path: Path) -> None:
    path = tmp_path / "input"
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(StatisticalAnalysisError, match="SHA-256 mismatch"):
        from src.analysis.stored_prediction_statistics import verify_hash

        verify_hash(path, "0" * 64, "protocol lock")


def test_no_locked_directory_write_in_synthetic_execution(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    marker = locked / "marker"
    marker.write_text("unchanged", encoding="utf-8")
    build_analysis(
        _paired(),
        _routing(),
        replicate_count=2,
        seed=42,
        expected_count=8,
        expected_support=np.array([2, 2, 2, 2]),
    )
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert list(locked.iterdir()) == [marker]

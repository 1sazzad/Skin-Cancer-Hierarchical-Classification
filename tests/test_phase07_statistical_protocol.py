from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis.statistical_protocol import (
    StatisticalProtocolError,
    frozen_linear_quantile,
    load_and_validate_protocol,
    validate_protocol,
)


CONFIG = Path("configs/analysis/phase07_paired_model_comparison.yaml")


def _protocol() -> dict[str, object]:
    return load_and_validate_protocol(CONFIG)


def test_primary_estimand_and_direction_are_frozen() -> None:
    protocol = _protocol()
    assert protocol["primary"]["estimand"] == "macro_f1_difference"
    assert protocol["primary"]["direction"] == "flat_minus_hierarchical"


def test_bootstrap_design_is_frozen_but_not_executed() -> None:
    bootstrap = _protocol()["bootstrap"]
    assert bootstrap["enabled_for_gate02"] is False
    assert bootstrap["method"] == "paired_stratified_with_replacement"
    assert bootstrap["pairing_unit"] == "image_id"
    assert bootstrap["strata"] == "ground_truth_class"
    assert bootstrap["preserve_support"] is True
    assert bootstrap["seed"] == 42
    assert bootstrap["replicate_count"] == 10000
    assert bootstrap["confidence_level"] == 0.95
    assert bootstrap["interval_method"] == "percentile"
    assert bootstrap["lower_quantile"] == 0.025
    assert bootstrap["upper_quantile"] == 0.975
    assert bootstrap["quantile_method"] == "linear"
    assert bootstrap["dtype"] == "float64"
    assert bootstrap["pre_quantile_rounding"] == "prohibited"
    assert bootstrap["deprecated_interpolation_fallback"] == "prohibited"
    assert bootstrap["zero_inclusion_bounds"] == "unrounded_machine_readable"


def test_classes_support_and_metric_policy_are_frozen() -> None:
    protocol = _protocol()
    assert protocol["classes"]["indices"] == [0, 1, 2, 3]
    assert protocol["classes"]["support"] == {
        "non_malignant": 2398,
        "melanoma": 678,
        "bcc": 498,
        "scc": 94,
    }
    assert protocol["bootstrap"]["zero_division"] == 0
    assert protocol["bootstrap"]["silently_drop_replicates"] is False
    assert protocol["bootstrap"]["non_finite_policy"] == "fail_closed"


def test_exact_two_sided_mcnemar_is_frozen_but_not_executed() -> None:
    mcnemar = _protocol()["mcnemar"]
    assert mcnemar["enabled_for_gate02"] is False
    assert mcnemar["method"] == "exact_binomial"
    assert mcnemar["alternative"] == "two_sided"
    assert mcnemar["discordant_pairs_only"] is True
    assert mcnemar["alpha"] == 0.05


def test_multiplicity_and_scc_policy_are_frozen() -> None:
    multiplicity = _protocol()["multiplicity"]
    assert multiplicity["per_class_f1_status"] == "exploratory_secondary"
    assert multiplicity["classwise_p_value_adjustment_if_added"] == "holm_bonferroni"
    assert multiplicity["scc_uncertainty_required"] is True


def test_no_checkpoint_or_inference_input_is_configured() -> None:
    protocol = _protocol()
    assert protocol["analysis_executed"] is False
    assert not {"checkpoint", "checkpoint_path", "model", "inference"}.intersection(
        protocol["input"]
    )
    assert protocol["routing_decomposition"]["checkpoint_reconstruction"] is False


def test_locked_hashes_match_gate01() -> None:
    hashes = _protocol()["input"]["locked_prediction_sha256"]
    assert hashes["hierarchical"] == "391557deb9a1aeb9b9f97edc9d3d38759e597d56b54bfdbab9ea7482451a221a"
    assert hashes["flat"] == "08b3462549210ed7f2330a687c37a6de4e013e00185fadc3167aa980995e497d"


def test_flat_archive_member_provenance_is_frozen() -> None:
    source = _protocol()["input"]["flat_archive_member_source"]
    assert source["archive_sha256"] == "b76762b53a35a8d9b0aa96621d78ea0e4421aa6e8052d068ffc10648a4e63e91"
    assert source["member_sha256"] == "08b3462549210ed7f2330a687c37a6de4e013e00185fadc3167aa980995e497d"
    assert source["reject_absolute_traversal_links_and_duplicates"] is True
    assert source["restore_to_locked_directory"] is False


@pytest.mark.parametrize(
    "values",
    [
        [-2.0, 0.0, 7.0],
        [-5.0, -1.0, 2.0, 9.0],
        [1.0, 1.0, 1.0, 4.0, 4.0],
        np.linspace(-3.0, 8.0, 10000, dtype=np.float64).tolist(),
    ],
)
@pytest.mark.parametrize("probability", [0.0, 0.025, 0.5, 0.975, 1.0])
def test_manual_linear_quantile_matches_numpy(
    values: list[float], probability: float
) -> None:
    expected = np.quantile(
        np.asarray(values, dtype=np.float64), probability, method="linear"
    )
    assert frozen_linear_quantile(values, probability) == pytest.approx(
        float(expected), rel=0.0, abs=4e-15
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_manual_quantile_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(StatisticalProtocolError, match="finite"):
        frozen_linear_quantile([0.0, value], 0.5)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("bootstrap", "seed", 7, "seed must be 42"),
        ("bootstrap", "replicate_count", 9999, "must be 10000"),
        ("bootstrap", "method", "ordinary", "method changed"),
        ("bootstrap", "quantile_method", "nearest", "must be linear"),
        ("bootstrap", "dtype", "float32", "must be float64"),
        ("bootstrap", "pre_quantile_rounding", "allowed", "must be prohibited"),
        ("bootstrap", "deprecated_interpolation_fallback", "allowed", "must be prohibited"),
        ("mcnemar", "method", "chi_square", "exact binomial"),
        ("primary", "direction", "hierarchical_minus_flat", "Primary estimand"),
    ],
)
def test_contradictory_fields_fail(
    section: str, field: str, value: object, message: str
) -> None:
    protocol = copy.deepcopy(_protocol())
    protocol[section][field] = value
    with pytest.raises(StatisticalProtocolError, match=message):
        validate_protocol(protocol)


def test_unknown_protocol_field_fails() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["unknown_future_choice"] = True
    with pytest.raises(StatisticalProtocolError, match="unknown"):
        validate_protocol(protocol)


def test_unknown_nested_protocol_field_fails() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["mcnemar"]["outcome_dependent_choice"] = True
    with pytest.raises(StatisticalProtocolError, match="unknown"):
        validate_protocol(protocol)


def test_missing_quantile_method_fails() -> None:
    protocol = copy.deepcopy(_protocol())
    del protocol["bootstrap"]["quantile_method"]
    with pytest.raises(StatisticalProtocolError, match="missing"):
        validate_protocol(protocol)


def test_incorrect_archive_member_configuration_fails() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["input"]["flat_archive_member_source"]["member_path"] = "other.csv"
    with pytest.raises(StatisticalProtocolError, match="member path"):
        validate_protocol(protocol)


def test_unknown_archive_nested_field_fails() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["input"]["flat_archive_member_source"]["unsafe_default"] = True
    with pytest.raises(StatisticalProtocolError, match="unknown"):
        validate_protocol(protocol)


def test_machine_readable_protocol_is_deterministic() -> None:
    first = json.dumps(_protocol(), indent=2, sort_keys=True) + "\n"
    second = json.dumps(_protocol(), indent=2, sort_keys=True) + "\n"
    assert first == second

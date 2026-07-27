"""Validation for the frozen Phase 07 statistical protocol."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import yaml


class StatisticalProtocolError(ValueError):
    """Raised when the frozen statistical protocol is invalid."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_CLASSES = ["non_malignant", "melanoma", "bcc", "scc"]
_EXPECTED_SUPPORT = {
    "non_malignant": 2398,
    "melanoma": 678,
    "bcc": 498,
    "scc": 94,
}
_TOP_LEVEL_KEYS = {
    "status",
    "gate",
    "analysis_executed",
    "input",
    "classes",
    "primary",
    "original_sample_point_estimates",
    "bootstrap",
    "primary_interval_interpretation",
    "mcnemar",
    "effect_measures",
    "multiplicity",
    "descriptive_analyses",
    "routing_decomposition",
    "claims",
    "reproducibility",
    "future_execution_fail_closed",
}
_SECTION_KEYS = {
    "original_sample_point_estimates": {
        "source",
        "model_metrics",
        "per_class_metrics",
        "paired_differences",
    },
    "primary_interval_interpretation": {
        "excludes_zero",
        "includes_zero",
        "prohibited_terms",
    },
    "mcnemar": {
        "enabled_for_gate02",
        "table",
        "method",
        "alternative",
        "discordant_pairs_only",
        "null_hypothesis",
        "alpha",
        "primary_asymptotic_result_allowed",
        "continuity_corrected_sensitivity_requires_authorization",
    },
    "effect_measures": {
        "primary",
        "secondary",
        "net_advantage_denominator",
        "discordant_odds_ratio",
        "zero_cell_policy",
        "haldane_anscombe",
        "odds_ratio_is_not_risk_ratio",
    },
    "multiplicity": {
        "secondary_model_level",
        "per_class_f1_status",
        "raw_per_class_intervals",
        "independent_unadjusted_class_significance_claims",
        "classwise_p_value_adjustment_if_added",
        "scc_uncertainty_required",
    },
    "descriptive_analyses": {
        "prediction_agreement",
        "scc",
        "confirmatory_status",
    },
    "routing_decomposition": {
        "allowed_only_if_stored_fields_and_semantics_proven",
        "allowed_categories",
        "infer_missing_states",
        "checkpoint_reconstruction",
        "fill_conditional_missing_values",
        "unsupported_decomposition_policy",
    },
    "claims": {"permitted", "prohibited"},
    "reproducibility": {"record", "deterministic_payload_timestamps_separate"},
}


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], section: str
) -> None:
    observed = set(payload)
    if observed != expected:
        raise StatisticalProtocolError(
            f"{section} keys differ: missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}."
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StatisticalProtocolError(message)


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the frozen protocol as a plain dictionary."""
    _require_exact_keys(protocol, _TOP_LEVEL_KEYS, "statistical_protocol")
    _require(protocol["status"] == "frozen_before_analysis", "Protocol is not frozen.")
    _require(protocol["gate"] == "phase07_gate02", "Gate must be phase07_gate02.")
    _require(protocol["analysis_executed"] is False, "Gate 2 cannot execute analysis.")
    for section, expected_keys in _SECTION_KEYS.items():
        _require(
            isinstance(protocol[section], Mapping),
            f"{section} must be a mapping.",
        )
        _require_exact_keys(protocol[section], expected_keys, section)

    input_spec = protocol["input"]
    _require_exact_keys(
        input_spec,
        {
            "paired_manifest_path",
            "paired_manifest_sha256",
            "sample_count",
            "identifier",
            "unit",
            "grouping_policy",
            "locked_prediction_sha256",
        },
        "input",
    )
    _require(input_spec["sample_count"] == 3668, "Sample count must be 3668.")
    _require(input_spec["identifier"] == "image_id", "Pairing identifier must be image_id.")
    _require(
        input_spec["unit"] == "paired_internal_test_image",
        "Analysis unit must be one paired internal-test image.",
    )
    hashes = input_spec["locked_prediction_sha256"]
    _require_exact_keys(hashes, {"hierarchical", "flat"}, "locked_prediction_sha256")
    for name, digest in {**hashes, "paired_manifest": input_spec["paired_manifest_sha256"]}.items():
        _require(
            isinstance(digest, str) and _SHA256.fullmatch(digest) is not None,
            f"{name} SHA-256 must contain 64 lowercase hexadecimal characters.",
        )
    _require(
        hashes["hierarchical"]
        == "391557deb9a1aeb9b9f97edc9d3d38759e597d56b54bfdbab9ea7482451a221a",
        "Hierarchical prediction hash differs from Gate 1.",
    )
    _require(
        hashes["flat"]
        == "08b3462549210ed7f2330a687c37a6de4e013e00185fadc3167aa980995e497d",
        "Flat prediction hash differs from Gate 1.",
    )

    classes = protocol["classes"]
    _require_exact_keys(classes, {"labels", "indices", "support"}, "classes")
    _require(classes["labels"] == _EXPECTED_CLASSES, "Class labels or order changed.")
    _require(classes["indices"] == [0, 1, 2, 3], "Class indices must be [0, 1, 2, 3].")
    _require(classes["support"] == _EXPECTED_SUPPORT, "Frozen class support changed.")

    primary = protocol["primary"]
    _require(
        primary
        == {
            "estimand": "macro_f1_difference",
            "direction": "flat_minus_hierarchical",
            "family_size": 1,
            "multiplicity_correction": "none",
        },
        "Primary estimand definition is contradictory or incomplete.",
    )

    bootstrap = protocol["bootstrap"]
    _require_exact_keys(
        bootstrap,
        {
            "enabled_for_gate02",
            "method",
            "pairing_unit",
            "strata",
            "preserve_support",
            "replicate_count",
            "seed",
            "confidence_level",
            "interval_method",
            "lower_quantile",
            "upper_quantile",
            "fixed_labels",
            "zero_division",
            "silently_drop_replicates",
            "non_finite_policy",
            "rationale",
            "authorized_model_metrics",
            "authorized_paired_differences",
        },
        "bootstrap",
    )
    _require(bootstrap["enabled_for_gate02"] is False, "Bootstrap must be disabled in Gate 2.")
    _require(bootstrap["method"] == "paired_stratified_with_replacement", "Bootstrap method changed.")
    _require(bootstrap["pairing_unit"] == "image_id", "Bootstrap must remain paired by image_id.")
    _require(bootstrap["strata"] == "ground_truth_class", "Bootstrap strata must be ground truth.")
    _require(bootstrap["preserve_support"] is True, "Bootstrap must preserve class support.")
    _require(bootstrap["replicate_count"] == 10000, "Replicate count must be 10000.")
    _require(bootstrap["seed"] == 42, "Bootstrap seed must be 42.")
    _require(bootstrap["confidence_level"] == 0.95, "Confidence level must be 0.95.")
    _require(bootstrap["interval_method"] == "percentile", "CI method must be percentile.")
    _require(
        bootstrap["lower_quantile"] == 0.025
        and bootstrap["upper_quantile"] == 0.975,
        "Percentile bounds must be 0.025 and 0.975.",
    )
    _require(bootstrap["fixed_labels"] == [0, 1, 2, 3], "Bootstrap labels changed.")
    _require(bootstrap["zero_division"] == 0, "zero_division must be 0.")
    _require(bootstrap["silently_drop_replicates"] is False, "Replicates cannot be dropped.")
    _require(bootstrap["non_finite_policy"] == "fail_closed", "Non-finite policy must fail closed.")

    mcnemar = protocol["mcnemar"]
    _require(mcnemar["enabled_for_gate02"] is False, "McNemar must be disabled in Gate 2.")
    _require(mcnemar["method"] == "exact_binomial", "McNemar method must be exact binomial.")
    _require(mcnemar["alternative"] == "two_sided", "McNemar must be two-sided.")
    _require(mcnemar["discordant_pairs_only"] is True, "McNemar must use discordant pairs.")
    _require(mcnemar["alpha"] == 0.05, "McNemar alpha must be 0.05.")
    _require(
        mcnemar["primary_asymptotic_result_allowed"] is False,
        "Asymptotic McNemar cannot be primary.",
    )

    multiplicity = protocol["multiplicity"]
    _require(
        multiplicity["per_class_f1_status"] == "exploratory_secondary",
        "Per-class F1 comparisons must remain exploratory.",
    )
    _require(
        multiplicity["classwise_p_value_adjustment_if_added"] == "holm_bonferroni",
        "Future class-wise p-values require Holm-Bonferroni.",
    )
    _require(
        multiplicity["independent_unadjusted_class_significance_claims"] is False,
        "Unadjusted class-wise significance claims are prohibited.",
    )
    _require(multiplicity["scc_uncertainty_required"] is True, "SCC uncertainty is required.")

    routing = protocol["routing_decomposition"]
    _require(routing["infer_missing_states"] is False, "Routing states cannot be inferred.")
    _require(routing["checkpoint_reconstruction"] is False, "Checkpoint reconstruction is prohibited.")
    _require(routing["fill_conditional_missing_values"] is False, "Conditional missing values cannot be filled.")

    forbidden_input_keys = {"checkpoint", "checkpoint_path", "model", "inference"}
    _require(
        not forbidden_input_keys.intersection(input_spec),
        "Checkpoint, model, or inference input is prohibited.",
    )
    return dict(protocol)


def load_and_validate_protocol(config_path: Path) -> dict[str, Any]:
    """Load the analysis YAML and validate its statistical protocol section."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("statistical_protocol"), dict
    ):
        raise StatisticalProtocolError("Missing statistical_protocol mapping.")
    return validate_protocol(payload["statistical_protocol"])

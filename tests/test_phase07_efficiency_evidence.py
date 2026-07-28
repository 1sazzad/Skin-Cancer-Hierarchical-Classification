from __future__ import annotations

from pathlib import Path

import torch

from src.analysis.efficiency_evidence import (
    PROHIBITED,
    bytes_to_mib,
    count_state_dictionary,
    generate_efficiency,
    mean_milliseconds,
    routing_compute,
    unavailable,
)
from src.analysis.stored_prediction_statistics import sha256_file


def test_conditional_compute_arithmetic() -> None:
    values = routing_compute(3668, 1799)
    assert values["stage2_not_invoked_count"] == 1869
    assert values["stage2_invocation_rate"] + values["stage2_bypass_rate"] == 1.0
    assert values["average_forward_passes_per_input"] == 1 + 1799 / 3668


def test_bytes_and_timing_derivations() -> None:
    assert bytes_to_mib(1_048_576) == 1.0
    assert mean_milliseconds(2.0, 4) == 500.0


def test_state_dictionary_count_excludes_optimizer_and_buffers() -> None:
    state = {
        "layer.weight": torch.zeros(2, 3),
        "layer.bias": torch.zeros(2),
        "bn.running_mean": torch.zeros(2),
        "bn.running_var": torch.zeros(2),
        "bn.num_batches_tracked": torch.zeros((), dtype=torch.long),
        "optimizer_state": {"tensor": torch.zeros(100)},
    }
    result = count_state_dictionary(state)
    assert result["parameter_elements"] == 8
    assert result["buffer_elements"] == 5


def test_unavailable_is_not_numeric_zero_and_claims_are_locked() -> None:
    row = unavailable("energy", "flat", "missing")
    assert row["value"] == "unavailable"
    assert row["evidence_grade"] == "U"
    assert "faster" in PROHIBITED


def test_generation_is_deterministic_and_preserves_prior_evidence(tmp_path: Path) -> None:
    source = Path("reports/phase07/generated")
    before = {path.name: sha256_file(path) for path in source.iterdir() if path.is_file()}
    first = generate_efficiency(Path("."), tmp_path / "one/generated", tmp_path / "one/reports")
    second = generate_efficiency(Path("."), tmp_path / "two/generated", tmp_path / "two/reports")
    assert len(first) == len(second) == 9
    for left, right in zip(first, second):
        assert left.name == right.name
        assert left.read_bytes() == right.read_bytes()
    after = {path.name: sha256_file(path) for path in source.iterdir() if path.is_file()}
    assert before == after

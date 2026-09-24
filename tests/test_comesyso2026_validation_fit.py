"""Synthetic/mocked COM-03 tests. No real checkpoints, images, or Modal calls."""
import ast
import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml

from src.evaluation import comesyso2026_probability_fusion as fusion
from src.evaluation import comesyso2026_validation_fit as fit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def shared():
    truth = np.tile(np.arange(4), 3)
    p1 = np.tile([0.8, 0.2], (12, 1))  # Every root predicts NM; Task 2 remains present.
    p2 = np.tile([0.2, 0.3, 0.5], (12, 1))
    return fusion.SharedPredictions(tuple(f"synthetic-{i}" for i in range(12)), truth,
                                    (truth != 0).astype(int), p1.argmax(1), p1,
                                    np.where(truth == 0, -1, truth - 1), p2.argmax(1), p2,
                                    (None,) * 12, 0.0)


@pytest.fixture
def lock():
    return {"seeds": list(fit.SEEDS), "sklearn_version": fit.sklearn.__version__,
            "protocol_path": fusion.PROTOCOL_PATH, "protocol_sha256": "a" * 64,
            "validation_manifest_path": fit.MANIFEST.as_posix(),
            "validation_manifest_sha256": "b" * 64,
            "checkpoints": [{"seed": s.seed, "historical_run_name": s.run_name,
                             "checkpoint_selected_epoch": s.epoch,
                             "checkpoint_path": s.path(Path('.')).as_posix(),
                             "checkpoint_sha256": str(s.seed) * 20} for s in fit.SPECS]}


@pytest.fixture
def protocol():
    return yaml.safe_load((ROOT / fusion.PROTOCOL_PATH).read_text(encoding="utf-8"))


def test_loader_explicit_validation(monkeypatch, tmp_path):
    dataset = Mock(return_value=object())
    loader = Mock()
    monkeypatch.setattr(fit, "ISIC2019HierarchicalDataset", dataset)
    monkeypatch.setattr(fit, "DataLoader", loader)
    fit.build_validation_loader(tmp_path)
    assert dataset.call_args.kwargs["split"] == "validation"
    assert dataset.call_args.kwargs["stage"] == "flat_four_class"
    assert loader.call_args.kwargs["shuffle"] is False
    assert loader.call_args.kwargs["drop_last"] is False
    transform = dataset.call_args.kwargs["transform"].transforms
    assert transform[1].size == [256] or transform[1].size == 256
    assert list(transform[2].size) == [224, 224]


@pytest.mark.parametrize("scope", [{"split": "internal_test"}, {"split": "test"},
                                        {"dataset": "hiba"}, {"stage": "stage_2"}])
def test_scope_rejected_before_dataset(monkeypatch, tmp_path, scope):
    dataset = Mock(side_effect=AssertionError("must not construct dataset"))
    monkeypatch.setattr(fit, "ISIC2019HierarchicalDataset", dataset)
    with pytest.raises(ValueError):
        fit.build_validation_loader(tmp_path, **scope)
    dataset.assert_not_called()


def test_frozen_protocol_and_features(protocol):
    fit.validate_protocol(protocol)
    assert fusion.FEATURE_ORDER == ("p_task1_malignant", "p_task2_melanoma", "p_task2_bcc", "p_task2_scc")
    assert [(s.seed, s.epoch) for s in fit.SPECS] == [(42, 10), (123, 1), (2026, 26)]


@pytest.mark.parametrize("seeds", [[42], [42, 123, 2027], [123, 42, 2026], [42, 123, 2026, 42]])
def test_exact_seeds(protocol, seeds):
    protocol["models"]["seeds"] = seeds
    with pytest.raises(ValueError):
        fit.validate_protocol(protocol)


def test_epoch_mismatch(protocol):
    protocol["models"]["selections"][0]["shared_hard_selected_epoch"] = 11
    with pytest.raises(ValueError):
        fit.validate_protocol(protocol)


def test_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        fit._inspect_checkpoint(tmp_path, fit.SPECS[0])


@pytest.mark.parametrize("field,value", [("epoch", 9), ("seed", 123)])
def test_checkpoint_identity_mismatch(field, value):
    spec = fit.SPECS[0]
    payload = {"model_state_dict": {}, "epoch": 10, "seed": 42}
    payload[field] = value
    summary = {"reportable_as_full_result": True, "run_name": spec.run_name, "best_epoch": 10, "seed": 42}
    with pytest.raises(ValueError):
        fit.validate_checkpoint_and_summary(payload, summary, spec)


@pytest.mark.parametrize("missing", ["seed", "config_metadata", "model_metadata"])
def test_missing_checkpoint_identity_fails(monkeypatch, tmp_path, missing):
    spec = fit.SPECS[0]
    config_relative = Path("configs/extensions/paul2026_swin/shared_hard_seed42.yaml")
    config = yaml.safe_load((ROOT / config_relative).read_text(encoding="utf-8"))
    (tmp_path / config_relative).parent.mkdir(parents=True)
    (tmp_path / config_relative).write_text(yaml.safe_dump(config), encoding="utf-8")
    path = spec.path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"mock checkpoint: never deserialized")
    fit._publish_json(path.parent / "run_summary.json", {
        "reportable_as_full_result": True, "run_name": spec.run_name, "best_epoch": 10, "seed": 42})
    payload = {"model_state_dict": {}, "epoch": 10, "seed": 42,
               "config_metadata": {"experiment": config["experiment"]}, "model_metadata": config["model"]}
    del payload[missing]
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    loader = Mock(side_effect=AssertionError("identity must fail before model construction"))
    monkeypatch.setattr(fit, "load_selected_model", loader)
    with pytest.raises(ValueError):
        fit._inspect_checkpoint(tmp_path, spec)
    loader.assert_not_called()


def test_duplicate_ids(shared):
    with pytest.raises(ValueError, match="Duplicate"):
        fit.validate_collection(replace(shared, sample_ids=(shared.sample_ids[0],) * 12))


@pytest.mark.parametrize("change", ["order", "targets", "count"])
def test_cross_seed_integrity(shared, change):
    cohort, _ = fit.validate_collection(shared)
    if change == "order":
        cohort["sample_ids"].reverse()
    elif change == "targets":
        cohort["targets"][0] = 1
    else:
        cohort["sample_ids"].pop()
        cohort["targets"].pop()
    with pytest.raises(ValueError, match="count/order/target"):
        fit.validate_collection(shared, expected_cohort=cohort)


@pytest.mark.parametrize("head", [1, 2])
@pytest.mark.parametrize("bad", ["nan", "inf", "range", "sum", "shape"])
def test_probability_contract(shared, head, bad):
    field = f"stage{head}_probabilities"
    array = getattr(shared, field).copy()
    if bad == "shape":
        array = array[:, :-1]
    else:
        array[0, 0] = {"nan": np.nan, "inf": np.inf, "range": 1.1, "sum": 0.0}[bad]
    with pytest.raises(ValueError):
        fit.validate_collection(replace(shared, **{field: array}))


@pytest.mark.parametrize("targets", [np.zeros(12, dtype=int), np.full(12, 9, dtype=int)])
def test_all_four_classes_required(shared, targets):
    with pytest.raises(ValueError):
        fit.validate_collection(replace(shared, flat_targets=targets))


def test_unmasked_collector_and_one_forward():
    class Frozen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, images):
            self.calls += 1
            return {"task1": torch.tensor([[2., 0.]]).repeat(4, 1),
                    "task2": torch.tensor([[1., 2., 3.]]).repeat(4, 1)}

    model = Frozen()
    shared = fusion.collect_shared_predictions(model, [{"image": torch.zeros(4, 3, 2, 2),
                                                        "target": torch.arange(4),
                                                        "image_id": ["a", "b", "c", "d"]}])
    _, features = fit.validate_collection(shared)
    assert model.calls == 1
    assert np.all(shared.stage1_predictions == 0)
    assert np.isfinite(shared.stage2_probabilities).all()
    np.testing.assert_array_equal(features[:, 1:], shared.stage2_probabilities)


def publish_fixture(tmp_path, shared, lock, seed=42):
    output = fit.output_directory(tmp_path)
    output.mkdir(parents=True, exist_ok=True)
    cohort, features = fit.validate_collection(shared)
    model = fusion.fit_fusion(features, shared.flat_targets, seed=seed, fitting_partition=fusion.FIT_PARTITION,
                              sample_ids=shared.sample_ids, target_sample_ids=shared.sample_ids)
    fit._publish_seed(output, seed, shared, model, lock, cohort)
    return output, model, features


def test_serialization_round_trip_and_provenance(tmp_path, shared, lock):
    output, model, features = publish_fixture(tmp_path, shared, lock)
    restored, metadata, _ = fit.validate_completed_seed(output / "seed42", 42, lock)
    np.testing.assert_array_equal(fusion.predict_fusion(model, features)[0],
                                  fusion.predict_fusion(restored, features)[0])
    assert metadata["checkpoint_selected_epoch"] == 10
    assert metadata["validation_class_counts"] == {str(c): 3 for c in range(4)}
    with pytest.raises(FileExistsError):
        fit._publish_seed(output, 42, shared, model, lock, fit.validate_collection(shared)[0])


@pytest.mark.parametrize("key", ["protocol_sha256", "validation_manifest_sha256"])
def test_completed_provenance_drift(tmp_path, shared, lock, key):
    output, _, _ = publish_fixture(tmp_path, shared, lock)
    lock[key] = "drift"
    with pytest.raises(ValueError):
        fit.validate_completed_seed(output / "seed42", 42, lock)


def test_partial_seed_fails(tmp_path, lock):
    directory = tmp_path / "seed42"
    directory.mkdir()
    (directory / "validation_probabilities.csv").write_text("partial", encoding="utf-8")
    with pytest.raises(ValueError, match="file set"):
        fit.validate_completed_seed(directory, 42, lock)


def test_corrupt_completed_seed_fails(tmp_path, shared, lock):
    output, _, _ = publish_fixture(tmp_path, shared, lock)
    (output / "seed42/fusion_model.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        fit.validate_completed_seed(output / "seed42", 42, lock)


def test_preflight_cannot_follow_outputs(monkeypatch, tmp_path):
    output = fit.output_directory(tmp_path)
    (output / "seed42").mkdir(parents=True)
    inspect = Mock(side_effect=AssertionError("too late"))
    monkeypatch.setattr(fit, "inspect_inputs", inspect)
    with pytest.raises(ValueError, match="before-inference"):
        fit.run_validation_fit_preflight(tmp_path)
    inspect.assert_not_called()


def test_lock_hash_drift_and_missing_lock(monkeypatch, tmp_path, lock):
    with pytest.raises(FileNotFoundError):
        fit.verify_preflight_lock(tmp_path)
    fit._publish_json(fit.output_directory(tmp_path) / fit.LOCK_NAME, lock)
    changed = copy.deepcopy(lock)
    changed["protocol_sha256"] = "changed"
    monkeypatch.setattr(fit, "inspect_inputs", lambda root: changed)
    with pytest.raises(ValueError, match="drift"):
        fit.verify_preflight_lock(tmp_path)


def mock_production(monkeypatch, tmp_path, lock, shared):
    output = fit.output_directory(tmp_path)
    fit._publish_json(output / fit.LOCK_NAME, lock)
    monkeypatch.setattr(fit, "verify_preflight_lock", lambda root: lock)
    monkeypatch.setattr(fit, "verify_input_bytes", lambda root, lock: None)
    dataset = SimpleNamespace(selected_frame={"image_id": np.asarray(shared.sample_ids)},
                              targets=shared.flat_targets.tolist())
    monkeypatch.setattr(fit, "build_validation_loader", lambda root: SimpleNamespace(dataset=dataset))
    monkeypatch.setattr(fit, "load_selected_model", lambda *a, **k: Mock())
    collector = Mock(return_value=shared)
    monkeypatch.setattr(fusion, "collect_shared_predictions", collector)
    return output, collector


def test_independent_models_and_completed_reuse(monkeypatch, tmp_path, lock, shared):
    output, collector = mock_production(monkeypatch, tmp_path, lock, shared)
    real_fit = fusion.fit_fusion
    fitted = []

    def tracked(*args, **kwargs):
        result = real_fit(*args, **kwargs)
        fitted.append(result)
        return result

    monkeypatch.setattr(fusion, "fit_fusion", tracked)
    summary = fit.run_validation_fit_production(tmp_path, device="cpu")
    assert [model.seed for model in fitted] == [42, 123, 2026]
    assert len({id(model.classifier) for model in fitted}) == 3
    assert collector.call_count == 3
    assert fit.run_validation_fit_production(tmp_path, device="cpu") == summary
    assert len(fitted) == 3
    assert collector.call_count == 3
    text = json.dumps(summary)
    for forbidden in ("accuracy", "macro_f1", "confusion_matrix", "ranking", "best_seed"):
        assert forbidden not in text
    assert (output / fit.COMPLETE_NAME).is_file()


def test_resume_skips_completed_seed(monkeypatch, tmp_path, lock, shared):
    publish_fixture(tmp_path, shared, lock, seed=42)
    _, collector = mock_production(monkeypatch, tmp_path, lock, shared)
    tracked = Mock(wraps=fusion.fit_fusion)
    monkeypatch.setattr(fusion, "fit_fusion", tracked)
    fit.run_validation_fit_production(tmp_path, device="cpu")
    assert [call.kwargs["seed"] for call in tracked.call_args_list] == [123, 2026]
    assert collector.call_count == 2


def test_partial_existing_seed_blocks_all_inference(monkeypatch, tmp_path, lock, shared):
    output, collector = mock_production(monkeypatch, tmp_path, lock, shared)
    (output / "seed2026").mkdir()
    with pytest.raises(ValueError):
        fit.run_validation_fit_production(tmp_path, device="cpu")
    collector.assert_not_called()


def test_hidden_abandoned_staging_is_ignored(monkeypatch, tmp_path, lock, shared):
    output, collector = mock_production(monkeypatch, tmp_path, lock, shared)
    (output / ".seed42.abandoned").mkdir()
    fit.run_validation_fit_production(tmp_path, device="cpu")
    assert collector.call_count == 3


def test_output_boundary(tmp_path):
    output = fit.output_directory(tmp_path)
    assert output == tmp_path / fit.OUTPUT
    assert output.is_relative_to(tmp_path / "results/extensions/comesyso2026_probability_fusion")
    assert not output.is_relative_to(tmp_path / "results/extensions/paul2026_swin")


def test_modal_output_namespace_without_filesystem_resolution(monkeypatch):
    # Pure POSIX paths keep this runtime-path check portable to Windows hosts.
    class LexicalPath(PurePosixPath):
        def absolute(self):
            return self

        def resolve(self):
            raise AssertionError("Output construction must not resolve the mount")

    monkeypatch.setattr(fit, "Path", LexicalPath)
    monkeypatch.setattr(fit, "OUTPUT", LexicalPath(fit.OUTPUT.as_posix()))
    output = fit.output_directory("/root/project")
    assert output.parent == PurePosixPath("/root/project/results/extensions/comesyso2026_probability_fusion")
    assert output.name == "validation_fit"


@pytest.mark.parametrize("alternative", [
    "results/extensions/paul2026_swin/validation_fit", "results/paper/validation_fit",
    "results/other/validation_fit", "../outside", "/outside",
    "results/extensions/comesyso2026_probability_fusion/../paul2026_swin/validation_fit",
])
def test_alternate_output_rejected(monkeypatch, tmp_path, alternative):
    monkeypatch.setattr(fit, "OUTPUT", Path(alternative))
    with pytest.raises(ValueError, match="frozen relative path"):
        fit.output_directory(tmp_path)


def test_project_root_traversal_rejected(tmp_path):
    with pytest.raises(ValueError, match="traversal"):
        fit.output_directory(tmp_path / ".." / "other")


def test_mounted_output_artifacts_and_lock(monkeypatch, tmp_path, lock):
    output = fit.output_directory(tmp_path)
    fit._publish_json(output / fit.LOCK_NAME, lock)
    mount = output.parent
    backing = tmp_path / "volume-backing"
    original_resolve = Path.resolve

    def mounted_resolve(path, *args, **kwargs):
        if path.is_relative_to(mount):
            return backing / path.relative_to(mount)
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", mounted_resolve)
    monkeypatch.setattr(fit, "inspect_inputs", lambda root: lock)
    assert fit.output_directory(tmp_path) == output
    assert fit._visible(output) == {fit.LOCK_NAME}
    assert fit.verify_preflight_lock(tmp_path) == lock

    def redirected_resolve(path, *args, **kwargs):
        if path == output / fit.LOCK_NAME:
            return tmp_path / "results/paul2026_swin/foreign.json"
        return mounted_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", redirected_resolve)
    with pytest.raises(ValueError, match="Redirected artifact"):
        fit._visible(output)
    with pytest.raises(ValueError, match="Redirected preflight lock"):
        fit.verify_preflight_lock(tmp_path)


def test_preflight_idempotent_without_loader_or_fit(monkeypatch, tmp_path, lock):
    monkeypatch.setattr(fit, "inspect_inputs", lambda root: lock)
    loader = Mock(side_effect=AssertionError("preflight must not construct loader"))
    fitter = Mock(side_effect=AssertionError("preflight must not fit"))
    monkeypatch.setattr(fit, "build_validation_loader", loader)
    monkeypatch.setattr(fusion, "fit_fusion", fitter)
    assert fit.run_validation_fit_preflight(tmp_path) == lock
    assert fit.run_validation_fit_preflight(tmp_path) == lock
    loader.assert_not_called()
    fitter.assert_not_called()


def test_incompatible_preflight_not_overwritten(monkeypatch, tmp_path, lock):
    monkeypatch.setattr(fit, "inspect_inputs", lambda root: lock)
    fit.run_validation_fit_preflight(tmp_path)
    original = (fit.output_directory(tmp_path) / fit.LOCK_NAME).read_bytes()
    changed = {**lock, "protocol_sha256": "different"}
    monkeypatch.setattr(fit, "inspect_inputs", lambda root: changed)
    with pytest.raises(ValueError):
        fit.run_validation_fit_preflight(tmp_path)
    assert (fit.output_directory(tmp_path) / fit.LOCK_NAME).read_bytes() == original


def test_no_production_loader_before_lock(monkeypatch, tmp_path):
    loader = Mock(side_effect=AssertionError("must verify lock first"))
    monkeypatch.setattr(fit, "build_validation_loader", loader)
    with pytest.raises(FileNotFoundError):
        fit.run_validation_fit_production(tmp_path, device="cpu")
    loader.assert_not_called()


def test_partial_final_fails_without_inference(monkeypatch, tmp_path, lock, shared):
    output, collector = mock_production(monkeypatch, tmp_path, lock, shared)
    (output / fit.SUMMARY_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Partial"):
        fit.run_validation_fit_production(tmp_path, device="cpu")
    collector.assert_not_called()


def test_resume_cross_seed_mismatch_before_fitting(monkeypatch, tmp_path, lock, shared):
    publish_fixture(tmp_path, shared, lock)
    _, collector = mock_production(monkeypatch, tmp_path, lock, shared)
    dataset = SimpleNamespace(selected_frame={"image_id": np.asarray(shared.sample_ids[::-1])},
                              targets=shared.flat_targets.tolist())
    monkeypatch.setattr(fit, "build_validation_loader", lambda root: SimpleNamespace(dataset=dataset))
    fitter = Mock(side_effect=AssertionError("must compare cohorts first"))
    monkeypatch.setattr(fusion, "fit_fusion", fitter)
    with pytest.raises(ValueError, match="Cross-seed"):
        fit.run_validation_fit_production(tmp_path, device="cpu")
    collector.assert_not_called()
    fitter.assert_not_called()


def test_summary_requires_all_seeds():
    with pytest.raises(ValueError, match="Exactly seeds"):
        fit.three_seed_summary({42: {}})


def test_byte_verification_detects_manifest_drift(tmp_path, lock):
    manifest = tmp_path / fit.MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text("synthetic manifest bytes", encoding="utf-8")
    protocol_path = tmp_path / fusion.PROTOCOL_PATH
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("synthetic protocol", encoding="utf-8")
    lock.update(source_sha256={}, checkpoints=[], protocol_sha256=fit.sha256_file(protocol_path),
                validation_manifest_sha256=fit.sha256_file(manifest))
    fit._publish_json(fit.output_directory(tmp_path) / fit.LOCK_NAME, lock)
    fit.verify_input_bytes(tmp_path, lock)
    manifest.write_text("changed bytes", encoding="utf-8")
    with pytest.raises(ValueError, match="drift"):
        fit.verify_input_bytes(tmp_path, lock)


@pytest.mark.parametrize("module_file,runtime_root,expected", [
    (PurePosixPath("/work/repo/scripts/comesyso2026/modal_comesyso2026_probability_fusion.py"),
     PurePosixPath("/root/project"), PurePosixPath("/work/repo")),
    (PureWindowsPath("D:/work/repo/scripts/comesyso2026/modal_comesyso2026_probability_fusion.py"),
     PureWindowsPath("/root/project"), PureWindowsPath("D:/work/repo")),
    (PurePosixPath("/root/modal_comesyso2026_probability_fusion.py"),
     PurePosixPath("/root/project"), PurePosixPath("/root/project")),
])
def test_modal_repository_root_resolution(module_file, runtime_root, expected):
    # Extract only the pure helper: no Modal import, image build, or remote call.
    path = ROOT / "scripts/comesyso2026/modal_comesyso2026_probability_fusion.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "resolve_repository_root")
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["resolve_repository_root"](module_file, runtime_root) == expected
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "REPOSITORY_ROOT" for t in n.targets))
    assert isinstance(assignment.value, ast.Call)
    assert assignment.value.func.id == "resolve_repository_root"


def test_modal_architecture_static():
    path = ROOT / "scripts/comesyso2026/modal_comesyso2026_probability_fusion.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    constants = {target.id: ast.literal_eval(node.value)
                 for node in tree.body if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Constant)
                 for target in node.targets if isinstance(target, ast.Name)}
    assert constants["APP_NAME"] == "comesyso2026-probability-fusion"
    assert constants["PROJECT_ROOT"] == "/root/project"
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    volume_calls = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "from_name"]
    assert {ast.literal_eval(n.args[0]) for n in volume_calls} == {
        "paul2026-swin-data", "paul2026-swin-results", "comesyso2026-probability-fusion-results"}
    assert all(next(ast.literal_eval(k.value) for k in n.keywords if k.arg == "create_if_missing") is False
               for n in volume_calls)
    assert '"/root/project/data/raw"' in text
    assert '"/root/project/results/extensions/paul2026_swin"' in text
    assert '"/root/project/results/extensions/comesyso2026_probability_fusion"' in text
    commits = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "commit"]
    assert len(commits) == 2
    assert all(isinstance(n.value, ast.Name) and n.value.id == "output_volume" for n in commits)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    preflight = functions["run_validation_fit_preflight"].decorator_list[0]
    production = functions["run_validation_fit_production"].decorator_list[0]
    assert "gpu" not in {k.arg for k in preflight.keywords}
    assert next(ast.literal_eval(k.value) for k in production.keywords if k.arg == "gpu") == "T4"


def test_launcher_only_two_modes(monkeypatch, capsys):
    functions = {}

    def lookup(app, name):
        assert app == "comesyso2026-probability-fusion"
        function = Mock()
        function.get_current_stats.return_value = SimpleNamespace(num_running_inputs=0, backlog=0)
        function.spawn.return_value = SimpleNamespace(object_id="mock-call")
        functions[name] = function
        return function

    modal = SimpleNamespace(App=lambda name: SimpleNamespace(local_entrypoint=lambda: lambda f: f),
                            Function=SimpleNamespace(from_name=lookup))
    monkeypatch.setitem(__import__("sys").modules, "modal", modal)
    spec = importlib.util.spec_from_file_location("com03_launcher", ROOT / "scripts/comesyso2026/launch_comesyso2026_validation_fit.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    assert set(launcher.FUNCTION_BY_MODE) == {"preflight", "fit"}
    for mode in ("isic", "hiba", "train", "internal_test"):
        with pytest.raises(ValueError):
            launcher.main(mode)
    for mode in ("preflight", "fit"):
        functions.clear()
        launcher.main(mode)
        functions[launcher.FUNCTION_BY_MODE[mode]].spawn.assert_called_once_with()
        assert sum(f.spawn.call_count for f in functions.values()) == 1
        output = capsys.readouterr().out
        assert "function_call_id: mock-call" in output
        assert "local_process_required: false" in output

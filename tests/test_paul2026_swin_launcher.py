"""Offline launcher contracts, using mocked data and no training epochs."""

from copy import deepcopy
import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml

from src.models.classification_backbone import SUPPORTED_CLASSIFICATION_ARCHITECTURES
from src.models.shared_three_task import SUPPORTED_SHARED_ARCHITECTURES
from src.training import baseline_experiment as baseline
from src.training import paul2026_swin as paul

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/extensions/paul2026_swin"


def config_path(system="flat", seed=42):
    return CONFIG_DIR / f"{system}_seed{seed}.yaml"


@pytest.mark.parametrize("system", ["flat", "shared_hard"])
@pytest.mark.parametrize("seed", [42, 123, 2026])
def test_six_configs_accepted_without_mutation(system, seed):
    config = paul.load_paul_config(config_path(system, seed))
    original = deepcopy(config)
    assert paul.validate_paul_config(config) == system
    assert config == original
    assert config["experiment"]["seed"] == seed
    assert config["experiment"]["run_name"] == f"paul2026_{system}_swin_t_seed{seed}"
    assert config["data"]["internal_test"] == paul.INTERNAL_TEST_POLICY


def test_historical_config_rejected():
    with pytest.raises(ValueError, match="PAUL"):
        paul.load_paul_config(ROOT / "configs/paper/flat/efficientnet_b0.yaml")


@pytest.mark.parametrize("seed", [0, 43, 124, 2027, "42", True])
def test_seed_set_is_exact(seed):
    assert paul.PAUL2026_SEEDS == (42, 123, 2026)
    config = paul.load_paul_config(config_path())
    config["experiment"]["seed"] = seed
    with pytest.raises(ValueError, match="seed"):
        paul.validate_paul_config(config)


@pytest.mark.parametrize("system", ["flat", "shared_hard"])
@pytest.mark.parametrize("field,value", [
    ("model.architecture", "efficientnet_b0"),
    ("model.pretrained_weights", "none"),
    ("model.dropout_probability", 0.3),
    ("training.optimizer.learning_rate", 0.1),
    ("training.scheduler.minimum_learning_rate", 0.001),
    ("data.internal_test.allowed", True),
    ("data.internal_test.construct_loader", True),
    ("data.internal_test.influence_selection", True),
])
def test_scientific_drift_rejected(system, field, value):
    config = paul.load_paul_config(config_path(system))
    target = config
    *parents, key = field.split(".")
    for parent in parents:
        target = target[parent]
    target[key] = value
    with pytest.raises(ValueError):
        paul.validate_paul_config(config)


@pytest.mark.parametrize("system,field", [
    ("flat", "split_manifest"), ("shared_hard", "isic2019_manifest"),
    ("shared_hard", "stage3_manifest"),
])
def test_partition_drift_rejected(system, field):
    config = paul.load_paul_config(config_path(system))
    config["data"][field] = "different_seed123.csv"
    with pytest.raises(ValueError):
        paul.validate_paul_config(config)


def test_shared_guard_normalizes_only_seed_and_architecture(monkeypatch):
    config = paul.load_paul_config(config_path("shared_hard", 2026))
    original = deepcopy(config)
    guard = Mock(wraps=paul.phase03._validate_frozen_config)
    monkeypatch.setattr(paul.phase03, "_validate_frozen_config", guard)
    paul.validate_paul_config(config)
    view = guard.call_args.args[0]
    expected = deepcopy(original)
    expected["experiment"]["seed"] = 42
    expected["model"]["architecture"] = "efficientnet_b0"
    assert view == expected
    assert config == original
    assert view is not config


def test_historical_registries_remain_frozen():
    for registry in (SUPPORTED_CLASSIFICATION_ARCHITECTURES, SUPPORTED_SHARED_ARCHITECTURES):
        assert len(registry) == 7
        assert "swin_t" not in registry


@pytest.mark.parametrize("injected", [False, True])
def test_baseline_runner_default_and_injected_builders(monkeypatch, tmp_path, injected):
    class StopBeforeTraining(Exception):
        pass

    historical_path = ROOT / "configs/paper/flat/efficientnet_b0.yaml"
    loader = Mock(wraps=paul.load_paul_config if injected else baseline.load_experiment_config)
    data_builder = Mock(return_value={})
    model_builder = Mock(side_effect=StopBeforeTraining)
    monkeypatch.setattr(baseline, "seed_everything", lambda seed: None)
    monkeypatch.setattr(baseline, "_environment_payload", lambda device: {})
    kwargs = {}
    if injected:
        kwargs = dict(config_loader=loader, model_builder=model_builder,
                      dataloader_builder=data_builder)
    else:
        monkeypatch.setattr(baseline, "load_experiment_config", loader)
        monkeypatch.setattr(baseline, "build_stage_dataloaders", data_builder)
        monkeypatch.setattr(baseline, "build_classification_model", model_builder)
    with pytest.raises(StopBeforeTraining):
        baseline.run_baseline_experiment(
            config_path() if injected else historical_path,
            project_root=tmp_path, output_root=tmp_path, device="cpu", **kwargs,
        )
    loader.assert_called_once()
    data_builder.assert_called_once()
    model_builder.assert_called_once()
    assert model_builder.call_args.args == (("swin_t" if injected else "efficientnet_b0"), 4)
    assert model_builder.call_args.kwargs["pretrained"] == "imagenet"


def test_flat_loader_never_constructs_internal_test(monkeypatch, tmp_path):
    calls = []

    class FakeDataset:
        def __init__(self, manifest, root, split, stage, transform, **kwargs):
            calls.append((split, stage, kwargs["verify_image_paths"]))

        def __len__(self):
            return 3

    monkeypatch.setattr(paul, "ISIC2019HierarchicalDataset", FakeDataset)
    loaders = paul.build_flat_dataloaders(
        tmp_path / "manifest.csv", tmp_path, "flat_four_class",
        config=paul.DataLoaderConfig(seed=123),
    )
    assert set(loaders) == {"train", "validation"}
    assert calls == [("train", "flat_four_class", True), ("validation", "flat_four_class", True)]
    assert loaders["train"].generator.initial_seed() == 123
    assert isinstance(loaders["train"].sampler, torch.utils.data.RandomSampler)
    assert isinstance(loaders["validation"].sampler, torch.utils.data.SequentialSampler)


@pytest.mark.parametrize("system", ["flat", "shared_hard"])
def test_offline_models_use_swin_builders(monkeypatch, system):
    builder = Mock(return_value=torch.nn.Linear(2, 2))
    if system == "flat":
        monkeypatch.setattr(paul, "build_swin_t_classification_model", builder)
        paul.build_flat_model("swin_t", 4, pretrained="none")
    else:
        original = paul.phase03.build_shared_three_task_efficientnet_b0
        monkeypatch.setattr(paul, "build_swin_t_shared_three_task_model", builder)
        with paul.shared_model_builder():
            paul.phase03.build_shared_three_task_efficientnet_b0(pretrained="none")
        assert paul.phase03.build_shared_three_task_efficientnet_b0 is original
    assert builder.call_args.kwargs["pretrained"] == "none"


def test_shared_builder_restored_on_failure():
    original = paul.phase03.build_shared_three_task_efficientnet_b0
    with pytest.raises(RuntimeError):
        with paul.shared_model_builder():
            raise RuntimeError("stop")
    assert paul.phase03.build_shared_three_task_efficientnet_b0 is original


@pytest.mark.parametrize("system", ["flat", "shared_hard"])
def test_preflight_offline_no_training_or_files(monkeypatch, tmp_path, system):
    config = paul.load_paul_config(config_path(system))
    for path in (paul.ISIC_MANIFEST, paul.STAGE3_MANIFEST):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.touch()
    before = set(tmp_path.rglob("*"))
    monkeypatch.setattr(paul, "seed_everything", lambda seed: None)
    def loader(count):
        return SimpleNamespace(dataset=range(count))
    monkeypatch.setattr(paul, "build_flat_dataloaders", Mock(return_value={
        "train": loader(17124), "validation": loader(3669),
    }))
    monkeypatch.setattr(paul, "build_shared_three_task_dataloaders", Mock(return_value=SimpleNamespace(
        train=loader(17718), validation_task1=loader(3669),
        validation_task2=loader(1271), validation_task3=loader(127),
        train_source_counts={"isic2019": 17124, "isic_stage03": 594, "combined": 17718},
    )))
    builder = Mock(return_value=torch.nn.Linear(2, 2))
    monkeypatch.setattr(paul, "build_swin_t_classification_model", builder)
    monkeypatch.setattr(paul, "build_swin_t_shared_three_task_model", builder)
    def forbidden(*args, **kwargs):
        pytest.fail("Preflight must not train, construct an optimizer, or save checkpoints")
    monkeypatch.setattr(paul, "run_baseline_experiment", forbidden)
    monkeypatch.setattr(paul.phase03, "_run_training", forbidden)
    monkeypatch.setattr(paul.phase03, "_build_components", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    report = paul.preflight(config, project_root=tmp_path)
    assert report["configured_pretrained_weights"] == "imagenet"
    assert report["instantiated_weights"] == "none"
    assert report["internal_test_loader_constructed"] is False
    assert report["parameters"] == 6
    assert builder.call_args.kwargs["pretrained"] == "none"
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("system", ["flat", "shared_hard"])
def test_full_run_dispatch_preserves_seed_identity_and_imagenet(monkeypatch, tmp_path, system):
    path = config_path(system, 2026)
    config = paul.load_paul_config(path)
    run = tmp_path / config["experiment"]["run_name"]
    seed_spy = Mock()
    monkeypatch.setattr(paul, "seed_everything", seed_spy)
    if system == "flat":
        runner = Mock()
        monkeypatch.setattr(paul, "run_baseline_experiment", runner)
        paul.run_paul_experiment(path, output_root=tmp_path)
        kwargs = runner.call_args.kwargs
        assert kwargs["config_loader"](path) == config
        assert kwargs["model_builder"] is paul.build_flat_model
        assert kwargs["dataloader_builder"] is paul.build_flat_dataloaders
        assert kwargs["run_directory"] == run
    else:
        builder = Mock(return_value=torch.nn.Linear(2, 2))
        monkeypatch.setattr(paul, "build_swin_t_shared_three_task_model", builder)
        def runner(actual, **kwargs):
            assert actual == config
            assert kwargs["run_directory"] == run
            paul.phase03.build_shared_three_task_efficientnet_b0(pretrained="imagenet")
        monkeypatch.setattr(paul.phase03, "_run_training", runner)
        paul.run_paul_experiment(path, output_root=tmp_path)
        seed_spy.assert_called_once_with(2026)
        assert yaml.safe_load((run / "resolved_config.yaml").read_text()) == config
        assert builder.call_args.kwargs["pretrained"] == "imagenet"


@pytest.mark.parametrize("system", ["flat", "shared_hard"])
def test_refuses_nonempty_run_directory(tmp_path, system):
    run = tmp_path / f"paul2026_{system}_swin_t_seed42"
    run.mkdir()
    sentinel = run / "existing.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        paul.run_paul_experiment(config_path(system), output_root=tmp_path)
    assert sentinel.read_text() == "keep"


def test_registry_stays_planned_and_paper_files_unchanged():
    from tests.test_paul2026_swin_protocol import test_paper_configs_and_registry_remain_unchanged
    test_paper_configs_and_registry_remain_unchanged()
    with (ROOT / "experiments/extension_registry.csv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["experiment_id"].startswith("paul2026_")]
    assert len(rows) == 6
    assert {row["status"] for row in rows} == {"PLANNED"}

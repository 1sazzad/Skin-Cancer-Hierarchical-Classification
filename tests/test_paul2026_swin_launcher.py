"""Offline launcher contracts, using mocked data and no training epochs."""

from copy import deepcopy
import ast
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
from src.training.engine import EpochResult
from scripts import train_paul2026_swin as train_cli
from scripts.train_paul2026_swin import positive_integer

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


def test_flat_bounded_arguments_are_forwarded_unchanged(monkeypatch, tmp_path):
    runner = Mock()
    monkeypatch.setattr(paul, "run_baseline_experiment", runner)
    paul.run_paul_experiment(
        config_path("flat"),
        output_root=tmp_path,
        epoch_limit=1,
        max_train_batches=10,
        max_validation_batches=5,
    )
    kwargs = runner.call_args.kwargs
    assert kwargs["epoch_limit"] == 1
    assert kwargs["max_train_batches"] == 10
    assert kwargs["max_validation_batches"] == 5


def test_flat_without_limits_keeps_full_run_controls_unset(monkeypatch, tmp_path):
    runner = Mock()
    monkeypatch.setattr(paul, "run_baseline_experiment", runner)
    paul.run_paul_experiment(config_path("flat"), output_root=tmp_path)
    kwargs = runner.call_args.kwargs
    assert kwargs["epoch_limit"] is None
    assert kwargs["max_train_batches"] is None
    assert kwargs["max_validation_batches"] is None


def test_paul_resume_reaches_baseline_for_existing_run(monkeypatch, tmp_path):
    runner = Mock()
    monkeypatch.setattr(paul, "run_baseline_experiment", runner)
    run_directory = tmp_path / "paul2026_flat_swin_t_seed42"
    run_directory.mkdir()
    (run_directory / "resolved_config.yaml").write_text("bootstrap\n")
    paul.run_paul_experiment(
        config_path("flat"), output_root=tmp_path, resume=True
    )
    assert runner.call_args.kwargs["resume"] is True
    assert runner.call_args.kwargs["run_directory"] == run_directory


def _resumable_test_components(monkeypatch, call_state, *, fail_on_call=None):
    def data_builder(*args, **kwargs):
        generator = torch.Generator().manual_seed(42)
        dataset = torch.utils.data.TensorDataset(
            torch.zeros(4, 1), torch.arange(4)
        )
        return {
            "train": torch.utils.data.DataLoader(
                dataset, batch_size=4, generator=generator, shuffle=True
            ),
            "validation": torch.utils.data.DataLoader(dataset, batch_size=4),
        }

    def model_builder(*args, **kwargs):
        return torch.nn.Linear(1, 4)

    def fake_epoch(model, dataloader, criterion, device, **kwargs):
        del dataloader, criterion, device
        call_state["calls"] += 1
        if fail_on_call == call_state["calls"]:
            raise RuntimeError("simulated preemption")
        optimizer = kwargs.get("optimizer")
        if optimizer is not None:
            parameter = next(model.parameters())
            optimizer.state[parameter]["step"] = torch.tensor(1.0)
            optimizer.state[parameter]["exp_avg"] = torch.zeros_like(parameter)
            optimizer.state[parameter]["exp_avg_sq"] = torch.zeros_like(parameter)
            optimizer.state[parameter]["marker"] = torch.tensor(
                [call_state["calls"]]
            )
            parameter.data.fill_(call_state["calls"])
        targets = torch.tensor([0, 1, 2, 3])
        predictions = targets.clone()
        return EpochResult(
            mean_loss=0.25,
            sample_count=4,
            targets=targets,
            predictions=predictions,
            probabilities=torch.nn.functional.one_hot(
                predictions, num_classes=4
            ).float(),
        )

    monkeypatch.setattr(baseline, "run_classification_epoch", fake_epoch)
    return data_builder, model_builder


def test_resumable_bootstrap_directory_restarts_at_epoch_one(monkeypatch, tmp_path):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / "resolved_config.yaml").write_text("bootstrap\n")
    (run_directory / "environment.json").write_text("{}\n")
    state = {"calls": 0}
    data_builder, model_builder = _resumable_test_components(monkeypatch, state)
    output = baseline.run_baseline_experiment(
        config_path("flat"),
        project_root=ROOT,
        output_root=tmp_path,
        device="cpu",
        epoch_limit=1,
        config_loader=paul.load_paul_config,
        model_builder=model_builder,
        dataloader_builder=data_builder,
        run_directory=run_directory,
        resume=True,
    )
    assert output.best_epoch == 1
    assert torch.load(
        run_directory / "last_checkpoint.pt", weights_only=False
    )["resume_state"]["completed_epoch"] == 1


def test_resumable_run_restores_epoch_state_without_duplicate_history(monkeypatch, tmp_path):
    run_directory = tmp_path / "run"
    state = {"calls": 0, "restored_parameter": None, "restored_marker": None}
    data_builder, model_builder = _resumable_test_components(monkeypatch, state)
    callback = Mock()
    baseline.run_baseline_experiment(
        config_path("flat"), project_root=ROOT, output_root=tmp_path, device="cpu",
        epoch_limit=1, config_loader=paul.load_paul_config,
        model_builder=model_builder, dataloader_builder=data_builder,
        run_directory=run_directory, resume=True, persist_callback=callback,
    )
    checkpoint_before = torch.load(
        run_directory / "last_checkpoint.pt", weights_only=False
    )
    assert callback.call_count == 1
    original_epoch = state["calls"]

    def observing_epoch(model, dataloader, criterion, device, **kwargs):
        del dataloader, criterion, device
        if kwargs.get("optimizer") is not None:
            state["restored_parameter"] = float(next(model.parameters()).flatten()[0])
            parameter = next(model.parameters())
            state["restored_marker"] = kwargs["optimizer"].state[parameter]["marker"].clone()
        return EpochResult(
            mean_loss=0.25, sample_count=4,
            targets=torch.tensor([0, 1, 2, 3]),
            predictions=torch.tensor([0, 1, 2, 3]),
            probabilities=torch.eye(4),
        )

    monkeypatch.setattr(baseline, "run_classification_epoch", observing_epoch)
    baseline.run_baseline_experiment(
        config_path("flat"), project_root=ROOT, output_root=tmp_path, device="cpu",
        epoch_limit=2, config_loader=paul.load_paul_config,
        model_builder=model_builder, dataloader_builder=data_builder,
        run_directory=run_directory, resume=True, persist_callback=callback,
    )
    checkpoint_after = torch.load(
        run_directory / "last_checkpoint.pt", weights_only=False
    )
    resume_state = checkpoint_after["resume_state"]
    assert original_epoch == 2
    assert state["restored_parameter"] == 1.0
    assert state["restored_marker"].item() == 1
    assert checkpoint_after["scheduler_state_dict"]["last_epoch"] == 2
    assert "scaler_state_dict" in resume_state
    assert resume_state["best_epoch"] == 1
    assert resume_state["epochs_without_improvement"] == 1
    assert resume_state["completed_epoch"] == 2
    assert [row["epoch"] for row in resume_state["history"]] == [1, 2]
    assert callback.call_count == 2
    assert "train_generator_state" in resume_state
    assert "python_rng_state" in resume_state
    assert "numpy_rng_state" in resume_state
    assert "torch_cpu_rng_state" in resume_state
    assert "cumulative_training_seconds" in resume_state
    assert checkpoint_before["resume_state"]["completed_epoch"] == 1


def test_resumable_failed_epoch_does_not_commit_or_publish_history(monkeypatch, tmp_path):
    run_directory = tmp_path / "run"
    state = {"calls": 0}
    data_builder, model_builder = _resumable_test_components(
        monkeypatch, state, fail_on_call=1
    )
    callback = Mock()
    with pytest.raises(RuntimeError, match="simulated preemption"):
        baseline.run_baseline_experiment(
            config_path("flat"), project_root=ROOT, output_root=tmp_path, device="cpu",
            epoch_limit=1, config_loader=paul.load_paul_config,
            model_builder=model_builder, dataloader_builder=data_builder,
            run_directory=run_directory, resume=True, persist_callback=callback,
        )
    assert callback.call_count == 0
    assert not (run_directory / "last_checkpoint.pt").exists()
    assert not (run_directory / "history.json").exists()


@pytest.mark.parametrize("mismatch", ["seed", "run_name", "config_identity"])
def test_resumable_checkpoint_identity_mismatch_is_rejected(
    monkeypatch, tmp_path, mismatch
):
    run_directory = tmp_path / "run"
    state = {"calls": 0}
    data_builder, model_builder = _resumable_test_components(monkeypatch, state)
    baseline.run_baseline_experiment(
        config_path("flat"), project_root=ROOT, output_root=tmp_path, device="cpu",
        epoch_limit=1, config_loader=paul.load_paul_config,
        model_builder=model_builder, dataloader_builder=data_builder,
        run_directory=run_directory, resume=True,
    )
    checkpoint_path = run_directory / "last_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if mismatch == "seed":
        checkpoint["resume_state"]["seed"] = 123
    elif mismatch == "run_name":
        checkpoint["resume_state"]["run_name"] = "wrong-run"
    else:
        checkpoint["resume_state"]["config_identity"]["experiment"]["run_name"] = (
            "wrong-run"
        )
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="identity"):
        baseline.run_baseline_experiment(
            config_path("flat"), project_root=ROOT, output_root=tmp_path, device="cpu",
            config_loader=paul.load_paul_config,
            model_builder=model_builder, dataloader_builder=data_builder,
            run_directory=run_directory, resume=True,
        )


@pytest.mark.parametrize("name", ["epoch_limit", "max_train_batches", "max_validation_batches"])
@pytest.mark.parametrize("value", [0, -1, True, "1"])
def test_smoke_limits_require_positive_integers(name, value):
    with pytest.raises(ValueError, match="positive integer"):
        paul.validate_smoke_limits(**{name: value})


def test_shared_hard_rejects_bounded_arguments(tmp_path):
    with pytest.raises(ValueError, match="Shared-Hard"):
        paul.run_paul_experiment(
            config_path("shared_hard"), output_root=tmp_path, epoch_limit=1
        )


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_cli_positive_integer_validation(value):
    with pytest.raises(Exception, match="positive integer"):
        positive_integer(value)


def test_cli_preflight_rejects_training_limits(monkeypatch):
    monkeypatch.setattr(
        train_cli.sys,
        "argv",
        [
            "train_paul2026_swin.py",
            "--config",
            str(config_path("flat")),
            "--preflight",
            "--epoch-limit",
            "1",
        ],
    )
    with pytest.raises(SystemExit):
        train_cli.main()


def test_modal_harness_is_fixed_flat_seed42_smoke_without_importing_modal():
    source_path = ROOT / "scripts/modal_paul2026_swin.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    image_source = source[source.index("image = ("):source.index("data_volume =")]
    assert image_source.index("pip_install_from_requirements") < image_source.index(
        "add_local_dir"
    )
    assert image_source.rstrip().endswith(")")
    assert "add_local_dir" not in image_source[image_source.index("add_local_dir") + 1:]
    remote_function = source[source.index("def run_flat_smoke") :]
    assert "import sys" in remote_function
    assert "if PROJECT_ROOT not in sys.path:" in remote_function
    assert "sys.path.insert(0, PROJECT_ROOT)" in remote_function
    assert remote_function.index("sys.path.insert") < remote_function.index(
        "from src.training.paul2026_swin"
    )
    assert 'PROJECT_ROOT = "/root/project"' in source
    assert 'remote_path=PROJECT_ROOT' in source
    assert '"/root/project/data/raw": data_volume' in source
    assert '"/root/project/results/extensions/paul2026_swin": results_volume' in source
    remote_path_lines = [
        line.strip()
        for line in source.splitlines()
        if "remote_path=" in line or "volumes={" in line
        or "data_volume" in line or "results_volume" in line
    ]
    assert all("\\" not in line for line in remote_path_lines)
    assert "Path(PROJECT_ROOT) / CONFIG_PATH" in source
    assert "project_root=Path(PROJECT_ROOT)" in source
    assert "output_root=Path(SMOKE_OUTPUT_ROOT)" in source
    assert 'CONFIG_PATH = "configs/extensions/paul2026_swin/flat_seed42.yaml"' in source
    assert "EPOCH_LIMIT = 1" in source
    assert "MAX_TRAIN_BATCHES = 50" in source
    assert "MAX_VALIDATION_BATCHES = 10" in source
    assert 'GPU = "T4"' in source
    assert "MAX_CONTAINERS = 1" in source
    assert "RETRIES = 0" in source
    assert "SINGLE_USE_CONTAINERS = True" in source
    assert "single_use_containers=SINGLE_USE_CONTAINERS" in source
    assert "scaledown_window=0" not in source
    assert "SCALEDOWN_WINDOW_SECONDS" not in source
    assert "ignore_runtime_data" in source
    assert "create_if_missing=False" in source
    assert "/root/project/data/raw/isic2019" in source
    assert "/root/project/data/raw/emb" in source
    assert "timing_benchmarks/flat_seed42_50train_10val" in source
    assert "web_endpoint" not in source
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"put", "reload"}
        for node in ast.walk(tree)
    )


def test_modal_harness_has_production_flat_seed42_function():
    source = (ROOT / "scripts/modal_paul2026_swin.py").read_text(encoding="utf-8")
    production_start = source.rfind("@app.function(", 0, source.index("def run_flat_production"))
    production = source[production_start:]
    assert 'CONFIG_PATH = "configs/extensions/paul2026_swin/flat_seed42.yaml"' in source
    assert "PRODUCTION_TIMEOUT_SECONDS = 14 * 60 * 60" in source
    assert 'PRODUCTION_OUTPUT_ROOT = "/root/project/results/extensions/paul2026_swin/runs"' in source
    assert "timeout=PRODUCTION_TIMEOUT_SECONDS" in production
    assert "gpu=GPU" in production
    assert "max_containers=MAX_CONTAINERS" in production
    assert "retries=RETRIES" in production
    assert "single_use_containers=SINGLE_USE_CONTAINERS" in production
    assert "epoch_limit=None" in production
    assert "max_train_batches=None" in production
    assert "max_validation_batches=None" in production
    assert "resume=True" in production
    assert "persist_callback=results_volume.commit" in production
    assert "Path(PRODUCTION_OUTPUT_ROOT)" in production
    assert 'print("mode: production")' in production
    assert 'print("resume_capable: true")' in production
    assert 'print("bounded_batches: false")' in production
    assert "stopped_early" in production
    assert "run_flat_production.remote()" not in source
    assert "run_flat_smoke.remote()" not in source
    assert "timing_benchmarks/flat_seed42_50train_10val" in source


def test_modal_ignore_callback_handles_relative_and_absolute_paths():
    source = (ROOT / "scripts/modal_paul2026_swin.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "ignore_runtime_data"
    )
    namespace = {"Path": Path, "REPOSITORY_ROOT": ROOT}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "modal_test", "exec"), namespace)
    ignore_runtime_data = namespace["ignore_runtime_data"]

    assert ignore_runtime_data(Path(".gitignore")) is False
    assert ignore_runtime_data(Path("requirements.txt")) is False
    assert ignore_runtime_data(Path(".git/config")) is True
    assert ignore_runtime_data(Path(".venv/Lib/site.py")) is True
    assert ignore_runtime_data(Path("data/raw/isic2019/x.jpg")) is True
    assert ignore_runtime_data(Path("results/foo.pt")) is True
    assert ignore_runtime_data(ROOT / "data/raw/isic2019/x.jpg") is True


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

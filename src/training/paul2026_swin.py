"""PAUL-2026 adapters around the frozen Flat and Phase-03 training loops."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

import scripts.train_phase03_shared_three_task as phase03
from src.data.dataloaders import DataLoaderConfig, _make_loader
from src.data.isic2019_dataset import ISIC2019HierarchicalDataset
from src.data.shared_three_task import build_shared_three_task_dataloaders
from src.data.transforms import build_eval_transform, build_train_transform
from src.models.transformer_extension import (
    build_swin_t_classification_model,
    build_swin_t_shared_three_task_model,
)
from src.training.baseline_experiment import run_baseline_experiment
from src.utils.reproducibility import seed_everything

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAUL2026_SEEDS = (42, 123, 2026)
ISIC_MANIFEST = "data/manifests/isic2019_train_val_test_split_seed42.csv"
STAGE3_MANIFEST = "data/manifests/emb_stage03_dermoscopic_split_seed42.csv"
DEFAULT_OUTPUT_ROOT = Path("results/extensions/paul2026_swin/runs")
INTERNAL_TEST_POLICY = {
    "allowed": False, "construct_loader": False, "influence_selection": False,
}


def validate_paul_config(config: dict[str, Any]) -> str:
    """Return the system after strict validation, without mutating the config."""
    try:
        experiment, model, data = (config[key] for key in ("experiment", "model", "data"))
        if experiment["research_stage"] != "paul2026_swin_extension":
            raise ValueError("Only PAUL extension configs are accepted.")
        seed = experiment["seed"]
        if type(seed) is not int or seed not in PAUL2026_SEEDS:
            raise ValueError(f"PAUL training seed must be one of {PAUL2026_SEEDS}.")
        if model["architecture"] != "swin_t":
            raise ValueError("PAUL architecture must be swin_t.")
        system = "shared_hard" if "task_losses" in config else "flat"
        if experiment["run_name"] != f"paul2026_{system}_swin_t_seed{seed}":
            raise ValueError("PAUL run_name must preserve the system and training seed.")
        if experiment["status"] != "ready_for_training":
            raise ValueError("PAUL config must be ready_for_training.")
        if data["internal_test"] != INTERNAL_TEST_POLICY or any(
            value is not False for value in data["internal_test"].values()
        ):
            raise ValueError("Internal test must remain prohibited.")

        if system == "shared_hard":
            frozen_view = deepcopy(config)
            frozen_view["experiment"]["seed"] = 42
            frozen_view["model"]["architecture"] = "efficientnet_b0"
            phase03._validate_frozen_config(frozen_view)
            template_path = "configs/paper/shared_hard/shared_three_task.yaml"
        else:
            template_path = "configs/paper/flat/efficientnet_b0.yaml"

        # Complement the historical guard with exact section comparisons, so
        # manifest paths, loss normalization, and loader details cannot drift.
        expected = yaml.safe_load((PROJECT_ROOT / template_path).read_text(encoding="utf-8"))
        expected["model"]["architecture"] = "swin_t"
        if system == "flat":
            expected["data"]["internal_test"] = INTERNAL_TEST_POLICY
        for section in expected:
            if section != "experiment" and config.get(section) != expected[section]:
                raise ValueError(f"PAUL scientific settings differ in {section}.")
        if set(config) != set(expected):
            raise ValueError("Unexpected PAUL config sections.")
        return system
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Malformed PAUL config: {error}") from error


def load_paul_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("PAUL config must be a mapping.")
    validate_paul_config(config)
    return config


def build_flat_dataloaders(
    manifest_path, project_root, stage, *, config=None,
    train_transform=None, eval_transform=None, verify_image_paths=True,
):
    """Construct exactly train and validation, never an internal-test dataset."""
    if stage != "flat_four_class":
        raise ValueError("PAUL Flat requires flat_four_class.")
    loader_config = config or DataLoaderConfig(batch_size=64)
    train_transform = train_transform or build_train_transform()
    eval_transform = eval_transform or build_eval_transform()
    return {
        split: _make_loader(
            ISIC2019HierarchicalDataset(
                manifest_path, project_root, split, stage,
                train_transform if split == "train" else eval_transform,
                # The paper template disables this check; PAUL always verifies
                # required training/validation paths before launching a run.
                verify_image_paths=True,
            ),
            split=split, config=loader_config,
        )
        for split in ("train", "validation")
    }


def build_flat_model(architecture, number_of_classes, *, pretrained="imagenet",
                     dropout_probability=0.2):
    """Adapt the historical builder signature without changing its registry."""
    if architecture != "swin_t":
        raise ValueError("PAUL architecture must be swin_t.")
    return build_swin_t_classification_model(
        number_of_classes, pretrained=pretrained, dropout_probability=dropout_probability,
    )


@contextmanager
def shared_model_builder():
    """Temporarily redirect Phase-03 construction; restore even on failure.

    This adapter is used by the single-run CLI, not concurrent in-process runs.
    """
    original = phase03.build_shared_three_task_efficientnet_b0
    phase03.build_shared_three_task_efficientnet_b0 = build_swin_t_shared_three_task_model
    try:
        yield
    finally:
        phase03.build_shared_three_task_efficientnet_b0 = original


def _loader_config(config):
    loader = config["loader"]
    return DataLoaderConfig(**{
        key: loader[key] for key in (
            "batch_size", "num_workers", "pin_memory", "persistent_workers",
            "prefetch_factor", "drop_last_train",
        )
    }, seed=config["experiment"]["seed"])


def validate_smoke_limits(
    *,
    epoch_limit: int | None = None,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
) -> None:
    """Validate optional bounded-run controls without changing their values."""
    for name, value in (
        ("epoch_limit", epoch_limit),
        ("max_train_batches", max_train_batches),
        ("max_validation_batches", max_validation_batches),
    ):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be a positive integer when supplied.")


def preflight(config, *, project_root=PROJECT_ROOT, device="cpu") -> dict[str, Any]:
    """Check local inputs and CPU model construction; no epochs or output files."""
    system = validate_paul_config(config)
    if str(device) != "cpu":
        raise ValueError("PAUL preflight requires --device cpu.")
    root = Path(project_root).expanduser().resolve()
    seed_everything(config["experiment"]["seed"])
    for relative in ([ISIC_MANIFEST] if system == "flat" else [ISIC_MANIFEST, STAGE3_MANIFEST]):
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
    loader_config = _loader_config(config)
    if system == "flat":
        loaders = build_flat_dataloaders(
            root / ISIC_MANIFEST, root, "flat_four_class", config=loader_config,
        )
        counts = {f"{split}_count": len(loader.dataset) for split, loader in loaders.items()}
        model = build_flat_model("swin_t", 4, pretrained="none")
    else:
        loaders = build_shared_three_task_dataloaders(
            root / ISIC_MANIFEST, root / STAGE3_MANIFEST, root,
            config=loader_config, verify_image_paths=True,
        )
        sources = config["data"]["training_sources"]
        if loaders.train_source_counts != {
            "isic2019": sources["isic2019_train"],
            "isic_stage03": sources["isic_stage3_train"],
            "combined": sources["combined_natural_pool"],
        }:
            raise ValueError("Resolved training-source counts differ from config.")
        counts = {
            "isic_train_count": loaders.train_source_counts["isic2019"],
            "stage3_train_count": loaders.train_source_counts["isic_stage03"],
            "combined_training_count": len(loaders.train.dataset),
            **{f"validation_task{task}_count": len(getattr(loaders, f"validation_task{task}").dataset)
               for task in (1, 2, 3)},
        }
        model = build_swin_t_shared_three_task_model(pretrained="none")
    return {
        "preflight": "PASS", "system": system, "device": "cpu",
        "seed": config["experiment"]["seed"], "run_name": config["experiment"]["run_name"],
        "configured_pretrained_weights": config["model"]["pretrained_weights"],
        "instantiated_weights": "none", **counts,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "internal_test_loader_constructed": False,
    }


def run_paul_experiment(
    config_path,
    *,
    project_root=PROJECT_ROOT,
    output_root=None,
    device="cpu",
    epoch_limit: int | None = None,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    resume: bool = False,
    persist_callback=None,
):
    """Execute a declared run through an existing loop; never use internal test."""
    validate_smoke_limits(
        epoch_limit=epoch_limit,
        max_train_batches=max_train_batches,
        max_validation_batches=max_validation_batches,
    )
    config = load_paul_config(config_path)
    system = validate_paul_config(config)
    limits = (epoch_limit, max_train_batches, max_validation_batches)
    if system == "shared_hard" and any(value is not None for value in limits):
        raise ValueError(
            "Shared-Hard bounded controls are not implemented yet."
        )
    if resume and any(value is not None for value in limits):
        raise ValueError("PAUL resume support is reserved for unbounded production runs.")
    root = Path(project_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve() if output_root is not None else root / DEFAULT_OUTPUT_ROOT
    run_directory = output / config["experiment"]["run_name"]
    if not resume and run_directory.exists() and any(run_directory.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_directory}")
    if system == "flat":
        return run_baseline_experiment(
            config_path, project_root=root, output_root=output, device=device,
            config_loader=load_paul_config, model_builder=build_flat_model,
            dataloader_builder=build_flat_dataloaders, run_directory=run_directory,
            epoch_limit=epoch_limit,
            max_train_batches=max_train_batches,
            max_validation_batches=max_validation_batches,
            resume=resume,
            persist_callback=persist_callback,
        )
    if resume:
        from src.training.shared_resume import prepare_resumable_directory, load_shared_resume

        run_directory = prepare_resumable_directory(run_directory, config)
        if (run_directory / "last_checkpoint.pt").exists():
            load_shared_resume(run_directory / "last_checkpoint.pt", config)
        resolved_device = phase03._resolve_device(device)
        seed_everything(config["experiment"]["seed"])
    else:
        resolved_device = phase03._resolve_device(device)
        seed_everything(config["experiment"]["seed"])
        run_directory = phase03._prepare_run_directory(run_directory)
    resume_options = {}
    if resume:
        resume_options["resume"] = True
    if persist_callback is not None:
        resume_options["persist_callback"] = persist_callback
    resolved_path = run_directory / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with shared_model_builder(), (run_directory / "logs/train_console.log").open("a" if resume else "w", encoding="utf-8") as log:
        tee = phase03.Tee(sys.stdout, log)
        with redirect_stdout(tee), redirect_stderr(tee):
            return phase03._run_training(
                config, config_path=resolved_path, project_root=root,
                run_directory=run_directory, device=resolved_device,
                **resume_options,
            )

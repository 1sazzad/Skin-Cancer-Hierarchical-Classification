"""Deploy PAUL Flat Swin-T functions; submit production via the dedicated launcher."""

from pathlib import Path
import time

import modal


APP_NAME = "paul2026-swin"
GPU = "T4"
MAX_CONTAINERS = 1
RETRIES = 0
TIMEOUT_SECONDS = 30 * 60
PRODUCTION_TIMEOUT_SECONDS = 14 * 60 * 60
SINGLE_USE_CONTAINERS = True
CONFIG_PATH = "configs/extensions/paul2026_swin/flat_seed42.yaml"
EPOCH_LIMIT = 1
MAX_TRAIN_BATCHES = 50
MAX_VALIDATION_BATCHES = 10
PROJECT_ROOT = "/root/project"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ISIC_DATA_PATH = "/root/project/data/raw/isic2019"
EMB_DATA_PATH = "/root/project/data/raw/emb"
SMOKE_OUTPUT_ROOT = (
    "/root/project/results/extensions/paul2026_swin/"
    "timing_benchmarks/flat_seed42_50train_10val"
)
PRODUCTION_OUTPUT_ROOT = "/root/project/results/extensions/paul2026_swin/runs"


def ignore_runtime_data(path: Path) -> bool:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(REPOSITORY_ROOT)
        except ValueError:
            return False

    excluded_roots = (
        Path(".git"),
        Path(".venv"),
        Path("runs"),
        Path("results"),
        Path("data") / "raw",
    )
    return any(
        candidate == excluded or candidate.is_relative_to(excluded)
        for excluded in excluded_roots
    )

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(REPOSITORY_ROOT / "requirements.txt"))
    .add_local_dir(
        str(REPOSITORY_ROOT),
        remote_path=PROJECT_ROOT,
        ignore=ignore_runtime_data,
    )
)
data_volume = modal.Volume.from_name("paul2026-swin-data", create_if_missing=False)
results_volume = modal.Volume.from_name(
    "paul2026-swin-results", create_if_missing=False
)


@app.function(
    image=image,
    gpu=GPU,
    max_containers=MAX_CONTAINERS,
    retries=RETRIES,
    timeout=TIMEOUT_SECONDS,
    single_use_containers=SINGLE_USE_CONTAINERS,
    volumes={
        "/root/project/data/raw": data_volume,
        "/root/project/results/extensions/paul2026_swin": results_volume,
    },
)
def run_flat_smoke():
    import sys

    import torch

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from src.training.paul2026_swin import run_paul_experiment

    for required in (Path(ISIC_DATA_PATH), Path(EMB_DATA_PATH)):
        if not required.is_dir():
            raise FileNotFoundError(
                f"Required Modal data directory is missing: {required}"
            )

    print("system: flat")
    print("backbone: swin_t")
    print("seed: 42")
    print(f"gpu: {GPU}")
    print(f"epoch_limit: {EPOCH_LIMIT}")
    print(f"max_train_batches: {MAX_TRAIN_BATCHES}")
    print(f"max_validation_batches: {MAX_VALIDATION_BATCHES}")

    started = time.perf_counter()
    outcome = run_paul_experiment(
        Path(PROJECT_ROOT) / CONFIG_PATH,
        project_root=Path(PROJECT_ROOT),
        output_root=Path(SMOKE_OUTPUT_ROOT),
        device="cuda",
        epoch_limit=EPOCH_LIMIT,
        max_train_batches=MAX_TRAIN_BATCHES,
        max_validation_batches=MAX_VALIDATION_BATCHES,
    )
    elapsed = time.perf_counter() - started
    memory_gib = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )
    print(f"run_directory: {outcome.run_directory}")
    print(f"best_epoch: {outcome.best_epoch}")
    print(f"best_validation_macro_f1: {outcome.best_validation_macro_f1}")
    print(f"elapsed_seconds: {elapsed:.3f}")
    print(f"max_memory_allocated_gib: {memory_gib:.3f}")


@app.function(
    image=image,
    gpu=GPU,
    max_containers=MAX_CONTAINERS,
    retries=RETRIES,
    timeout=PRODUCTION_TIMEOUT_SECONDS,
    single_use_containers=SINGLE_USE_CONTAINERS,
    volumes={
        "/root/project/data/raw": data_volume,
        "/root/project/results/extensions/paul2026_swin": results_volume,
    },
)
def run_flat_production():
    import sys

    import torch

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from src.training.paul2026_swin import load_paul_config, run_paul_experiment

    for required in (Path(ISIC_DATA_PATH), Path(EMB_DATA_PATH)):
        if not required.is_dir():
            raise FileNotFoundError(
                f"Required Modal data directory is missing: {required}"
            )

    config_path = Path(PROJECT_ROOT) / CONFIG_PATH
    config = load_paul_config(config_path)
    training = config["training"]
    print("mode: production")
    print("resume_capable: true")
    print("system: flat")
    print("backbone: swin_t")
    print("seed: 42")
    print(f"gpu: {GPU}")
    print(f"max_epochs: {training['epochs']}")
    print(f"early_stopping_patience: {training['early_stopping_patience']}")
    print("bounded_batches: false")

    started = time.perf_counter()
    outcome = run_paul_experiment(
        config_path,
        project_root=Path(PROJECT_ROOT),
        output_root=Path(PRODUCTION_OUTPUT_ROOT),
        device="cuda",
        epoch_limit=None,
        max_train_batches=None,
        max_validation_batches=None,
        resume=True,
        persist_callback=results_volume.commit,
    )
    elapsed = time.perf_counter() - started
    memory_gib = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )
    print(f"run_directory: {outcome.run_directory}")
    print(f"best_epoch: {outcome.best_epoch}")
    print(f"best_validation_macro_f1: {outcome.best_validation_macro_f1}")
    print(f"stopped_early: {outcome.stopped_early}")
    print(f"elapsed_seconds: {elapsed:.3f}")
    print(f"max_memory_allocated_gib: {memory_gib:.3f}")


@app.local_entrypoint()
def main():
    raise SystemExit(
        "Production requires the deployed app. First run: "
        "modal deploy scripts/modal_paul2026_swin.py\n"
        "Then submit with: modal run scripts/launch_paul2026_swin_production.py"
    )

"""Run the bounded PAUL Flat Swin-T seed-42 smoke test with Modal."""

from pathlib import Path
import time

import modal


APP_NAME = "paul2026-swin"
GPU = "T4"
MAX_CONTAINERS = 1
RETRIES = 0
TIMEOUT_SECONDS = 30 * 60
SCALEDOWN_WINDOW_SECONDS = 0
CONFIG_PATH = "configs/extensions/paul2026_swin/flat_seed42.yaml"
EPOCH_LIMIT = 1
MAX_TRAIN_BATCHES = 10
MAX_VALIDATION_BATCHES = 5
PROJECT_ROOT = Path("/root/project")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ISIC_DATA_PATH = "/root/project/data/raw/isic2019"
EMB_DATA_PATH = "/root/project/data/raw/emb"
SMOKE_OUTPUT_ROOT = PROJECT_ROOT / "results/extensions/paul2026_swin/smoke_runs"


def ignore_runtime_data(path: Path) -> bool:
    relative = path.relative_to(REPOSITORY_ROOT)
    return relative.parts[:1] in {(".git",), (".venv",), ("runs",), ("results",)} or (
        relative.parts[:2] == ("data", "raw")
    )

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_dir(
        str(REPOSITORY_ROOT),
        remote_path=str(PROJECT_ROOT),
        ignore=ignore_runtime_data,
    )
    .pip_install_from_requirements(str(REPOSITORY_ROOT / "requirements.txt"))
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
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    volumes={
        str(PROJECT_ROOT / "data/raw"): data_volume,
        str(PROJECT_ROOT / "results/extensions/paul2026_swin"): results_volume,
    },
)
def run_flat_smoke():
    import torch

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
        PROJECT_ROOT / CONFIG_PATH,
        project_root=PROJECT_ROOT,
        output_root=SMOKE_OUTPUT_ROOT,
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


@app.local_entrypoint()
def main():
    run_flat_smoke.remote()
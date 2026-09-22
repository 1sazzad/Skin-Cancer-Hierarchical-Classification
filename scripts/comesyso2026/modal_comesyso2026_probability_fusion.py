"""Dedicated COM-03 deployment; importing defines functions but runs no study."""
from pathlib import Path

import modal

APP_NAME = "comesyso2026-probability-fusion"
PROJECT_ROOT = "/root/project"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def ignore_runtime_data(path):
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(REPOSITORY_ROOT)
        except ValueError:
            return False
    excluded = (Path(".git"), Path(".venv"), Path("runs"), Path("results"),
                Path("data/raw"), Path("data/external"))
    return any(candidate == p or candidate.is_relative_to(p) for p in excluded)


app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(REPOSITORY_ROOT / "requirements.txt"))
    .add_local_dir(str(REPOSITORY_ROOT), remote_path=PROJECT_ROOT, ignore=ignore_runtime_data)
)
data_volume = modal.Volume.from_name("paul2026-swin-data", create_if_missing=False)
checkpoint_volume = modal.Volume.from_name("paul2026-swin-results", create_if_missing=False)
output_volume = modal.Volume.from_name("comesyso2026-probability-fusion-results", create_if_missing=False)
INPUT_CHECKPOINT_MOUNT = "/root/project/results/extensions/paul2026_swin"
OUTPUT_MOUNT = "/root/project/results/extensions/comesyso2026_probability_fusion"


@app.function(
    image=image, max_containers=1, retries=0, timeout=60 * 60, single_use_containers=True,
    volumes={INPUT_CHECKPOINT_MOUNT: checkpoint_volume, OUTPUT_MOUNT: output_volume},
)
def run_validation_fit_preflight():
    """CPU-only; no raw-image mount, validation loader, inference, or fitting."""
    import sys
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from src.evaluation.comesyso2026_validation_fit import run_validation_fit_preflight as preflight
    return preflight(PROJECT_ROOT, persist_callback=output_volume.commit)


@app.function(
    image=image, gpu="T4", max_containers=1, retries=0, timeout=4 * 60 * 60,
    single_use_containers=True,
    volumes={"/root/project/data/raw": data_volume,
             INPUT_CHECKPOINT_MOUNT: checkpoint_volume, OUTPUT_MOUNT: output_volume},
)
def run_validation_fit_production():
    import sys
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from src.evaluation.comesyso2026_validation_fit import run_validation_fit_production as fit
    return fit(PROJECT_ROOT, device="cuda", persist_callback=output_volume.commit)


@app.local_entrypoint()
def main():
    raise SystemExit("Use the dedicated deployed-app launcher with mode preflight or fit.")

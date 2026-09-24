"""Deploy COM-04 on the existing app without changing COM-03 locked source.

Deploy this module instead of the original module to register both phases.
Submit only one launcher at a time; cross-function busy checks are advisory.
"""
import sys
from pathlib import Path

# Support local source layout and Modal's flattened module layout.
_directory = Path(__file__).resolve().parent
_repository = (_directory.parent.parent if _directory.name == "comesyso2026"
               else Path("/root/project"))
if str(_repository) not in sys.path:
    sys.path.insert(0, str(_repository))

from scripts.comesyso2026.modal_comesyso2026_probability_fusion import (
    app, image, PROJECT_ROOT, INPUT_CHECKPOINT_MOUNT, OUTPUT_MOUNT,
    data_volume, checkpoint_volume, output_volume,
)


@app.function(
    image=image, max_containers=1, retries=0, timeout=60 * 60,
    single_use_containers=True,
    volumes={INPUT_CHECKPOINT_MOUNT: checkpoint_volume, OUTPUT_MOUNT: output_volume},
)
def run_internal_isic_preflight():
    """CPU only, no raw-data mount; persist lock before GPU submission."""
    from src.evaluation.comesyso2026_internal_isic import run_internal_isic_preflight as preflight
    return preflight(PROJECT_ROOT, persist_callback=output_volume.commit)


@app.function(
    image=image, gpu="T4", max_containers=1, retries=0, timeout=4 * 60 * 60,
    single_use_containers=True,
    volumes={"/root/project/data/raw": data_volume,
             INPUT_CHECKPOINT_MOUNT: checkpoint_volume, OUTPUT_MOUNT: output_volume},
)
def run_internal_isic_production():
    from src.evaluation.comesyso2026_internal_isic import run_internal_isic_production as evaluate
    return evaluate(PROJECT_ROOT, device="cuda", persist_callback=output_volume.commit)

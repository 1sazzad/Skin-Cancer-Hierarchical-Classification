"""COM-05 additions to the existing CoMeSySo app; COM-03/04 stay unchanged.

Deploy this module to register all three phases. Submit one launcher at a time.
"""
import sys
from pathlib import Path

_directory = Path(__file__).resolve().parent
_repository = (_directory.parent.parent if _directory.name == "comesyso2026"
               else Path("/root/project"))
if str(_repository) not in sys.path:
    sys.path.insert(0, str(_repository))

from scripts.comesyso2026.modal_comesyso2026_internal_isic import (
    app, image, PROJECT_ROOT, INPUT_CHECKPOINT_MOUNT, OUTPUT_MOUNT,
    checkpoint_volume, output_volume,
)
from scripts.comesyso2026.modal_comesyso2026_probability_fusion import REPOSITORY_ROOT

# The source image deliberately excludes raw external data. Ship only the frozen
# manifest with the image. Real HIBA bytes live in a dedicated Modal volume
# mounted outside PROJECT_ROOT because the historical path-safety check requires
# staged manifest paths to resolve inside PROJECT_ROOT.
HIBA_MANIFEST = "data/external/hiba/manifests/hiba_external_dermoscopic_4class_final.csv"
HIBA_SOURCE_MOUNT = "/mnt/comesyso2026-hiba-data"
hiba_volume = modal.Volume.from_name("comesyso2026-hiba-data", create_if_missing=False)
hiba_image = image.add_local_file(str(REPOSITORY_ROOT / HIBA_MANIFEST),
                                  remote_path=f"{PROJECT_ROOT}/{HIBA_MANIFEST}")


@app.function(
    image=hiba_image, max_containers=1, retries=0, timeout=2 * 60 * 60,
    single_use_containers=True,
    volumes={HIBA_SOURCE_MOUNT: hiba_volume,
             INPUT_CHECKPOINT_MOUNT: checkpoint_volume, OUTPUT_MOUNT: output_volume},
)
def run_external_hiba_preflight():
    """CPU-only checkpoint/cohort hashing; no image decoding or forward passes."""
    from src.evaluation.comesyso2026_external_hiba import (
        stage_hiba_images, run_external_hiba_preflight as preflight,
    )
    stage_hiba_images(PROJECT_ROOT, HIBA_SOURCE_MOUNT)
    return preflight(PROJECT_ROOT, persist_callback=output_volume.commit)


@app.function(
    image=hiba_image, gpu="T4", max_containers=1, retries=0, timeout=4 * 60 * 60,
    single_use_containers=True,
    volumes={"/root/project/data/raw": data_volume,
             INPUT_CHECKPOINT_MOUNT: checkpoint_volume, OUTPUT_MOUNT: output_volume},
)
def run_external_hiba_inference():
    """Publish raw paired inference, commit each seed, return without statistics."""
    from src.evaluation.comesyso2026_external_hiba import (
        stage_hiba_images, run_external_hiba_inference as infer,
    )
    stage_hiba_images(PROJECT_ROOT, HIBA_SOURCE_MOUNT)
    return infer(PROJECT_ROOT, device="cuda", persist_callback=output_volume.commit)


@app.function(
    image=hiba_image, max_containers=1, retries=0, timeout=12 * 60 * 60,
    single_use_containers=True,
    volumes={OUTPUT_MOUNT: output_volume},
)
def run_external_hiba_analysis():
    """CPU-only persisted-prediction analysis; no raw-data or checkpoint mount."""
    from src.evaluation.comesyso2026_external_hiba import run_external_hiba_analysis as analyze
    return analyze(PROJECT_ROOT, persist_callback=output_volume.commit)

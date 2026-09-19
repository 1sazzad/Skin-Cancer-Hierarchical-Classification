"""Offline protocol checks; no models, datasets, or training are executed."""

import copy
import csv
import hashlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/extensions/paul2026_swin"
SEEDS = {42, 123, 2026}
ISIC = "data/manifests/isic2019_train_val_test_split_seed42.csv"
STAGE3 = "data/manifests/emb_stage03_dermoscopic_split_seed42.csv"
INTERNAL_TEST = dict(allowed=False, construct_loader=False, influence_selection=False)
TEMPLATES = {
    "flat": "configs/paper/flat/efficientnet_b0.yaml",
    "shared_hard": "configs/paper/shared_hard/shared_three_task.yaml",
}


def read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def configs(system):
    return [read_yaml(CONFIG_DIR / f"{system}_seed{seed}.yaml") for seed in sorted(SEEDS)]


def test_exactly_six_configs():
    assert {path.name for path in CONFIG_DIR.glob("*.yaml")} == {
        f"{system}_seed{seed}.yaml" for system in TEMPLATES for seed in SEEDS
    }


@pytest.mark.parametrize("system", TEMPLATES)
def test_seeds_architecture_and_fixed_partitions(system):
    documents = configs(system)
    assert {doc["experiment"]["seed"] for doc in documents} == SEEDS
    for doc in documents:
        assert doc["model"]["architecture"] == "swin_t"
        assert doc["model"]["pretrained_weights"] == "imagenet"
        if system == "flat":
            assert doc["data"]["split_manifest"] == ISIC
        else:
            assert doc["data"]["isic2019_manifest"] == ISIC
            assert doc["data"]["stage3_manifest"] == STAGE3
        assert doc["data"]["internal_test"] == INTERNAL_TEST


@pytest.mark.parametrize("system", TEMPLATES)
def test_only_training_seed_and_run_identity_vary(system):
    normalized = []
    for doc in configs(system):
        doc = copy.deepcopy(doc)
        experiment = doc["experiment"]
        seed = experiment.pop("seed")
        assert experiment.pop("run_name") == f"paul2026_{system}_swin_t_seed{seed}"
        assert experiment.pop("output_root") == "runs/extensions/paul2026_swin/"
        normalized.append(doc)
    assert normalized[0] == normalized[1] == normalized[2]


@pytest.mark.parametrize("system", TEMPLATES)
def test_scientific_semantics_match_paper_template(system):
    # Compare the entire config, including every loss weight, sampling option,
    # loader setting, preprocessing choice, and checkpoint-selection setting.
    template = read_yaml(ROOT / TEMPLATES[system])
    for doc in configs(system):
        expected = copy.deepcopy(template)
        expected["model"]["architecture"] = "swin_t"
        expected["experiment"].update(
            status="ready_for_training", research_stage="paul2026_swin_extension",
            seed=doc["experiment"]["seed"],
            run_name=doc["experiment"]["run_name"],
            output_root="runs/extensions/paul2026_swin/",
        )
        if system == "flat":
            expected["experiment"]["model"] = "swin_t"
            expected["data"]["internal_test"] = INTERNAL_TEST
        assert doc == expected


def test_extension_registry_preserves_existing_rows_and_adds_six_planned_runs():
    text = (ROOT / "experiments/extension_registry.csv").read_text(encoding="utf-8")
    # Preserve the original header and rows, allowing Git's CRLF/LF conversion.
    prefix = "".join(text.splitlines(keepends=True)[:3])
    assert hashlib.sha256(prefix.encode("utf-8")).hexdigest() == (
        "2a66641f972275b447425e450bf51aff70de4eac2f3a5b6b1e54f415b4bd84fa"
    )
    rows = list(csv.DictReader(text.splitlines()))
    expected = {
        f"paul2026_{system}_swin_t_seed{seed}": (system, seed)
        for system in TEMPLATES for seed in SEEDS
    }
    added = rows[2:]
    assert len(added) == 6
    assert {row["experiment_id"] for row in added} == set(expected)
    for row in added:
        system, seed = expected[row["experiment_id"]]
        assert row["status"] == "PLANNED"
        assert row["backbone"] == "swin_t"
        assert int(row["seed"]) == seed
        assert row["system"] == {"flat": "Flat", "shared_hard": "Shared-Hard"}[system]
        assert row["config"] == f"configs/extensions/paul2026_swin/{system}_seed{seed}.yaml"
        assert read_yaml(ROOT / row["config"])["experiment"]["run_name"] == row["experiment_id"]
        for field in ("validation_metric", "internal_test_macro_f1", "external_hiba_macro_f1",
                      "checkpoint_sha256", "metrics_path", "predictions_path"):
            assert row[field] == ""


# Frozen text hashes at 74f95a5; newline normalization permits Windows/Linux.
PAPER_SHA256 = {'configs/paper/evaluation/phase06d_internal_isic.yaml': '2b42d5e3af33b3be49b1c2330154e34db5629b7eb135222e0ac58fe3d7c1a1dd',
 'configs/paper/evaluation/phase06f_hiba_external.yaml': 'd938c51b9e710e9faf10dad86a0d6db16d4ae2b9029c9230abddc929f31ceaa5',
 'configs/paper/evaluation/phase07_statistical_comparison.yaml': 'b2e2c38ab23cc442d11e33b30b08f12009e72f4ec933ca984f192cba4eaba1eb',
 'configs/paper/flat/densenet121.yaml': '44877ded7097a75d7a8b4b33a6a117d1358283479b4e8052f756b2538f9855b0',
 'configs/paper/flat/densenet169.yaml': '3267d691fd6fa86cb46c065b4677f0038a64a9f5f974992229cebb23597c67b7',
 'configs/paper/flat/efficientnet_b0.yaml': '4bcd7d1ef4ac52d4fefd6aa0c298de81af3d623de474825cc3d427a0abcf19d3',
 'configs/paper/flat/efficientnet_b2.yaml': '40a3e433de0f7231068ba9b5cbb1f26f12bd61d003a76224d9e909bac4bc7df5',
 'configs/paper/flat/efficientnet_b3.yaml': 'e12fcc225f6c0743c8a60ae6c46fb901eb047731e1025ffca9aed37a17e26a64',
 'configs/paper/flat/mobilenet_v3_large.yaml': 'a8e8d411553cc88b4d2aa4cb6bf290f4fe67d2855e9d32dc9b8ba0a8681a2ec6',
 'configs/paper/flat/resnet50.yaml': '4571e1057d36a0b9506519c3959f911fc7e20354a24528ff2e7a788b06a59e52',
 'configs/paper/shared_hard/shared_three_task.yaml': '40c08bc0c89d9a1244b97ad4cd3d828ef86ae7145f7e6d13609211e63f60de8a',
 'experiments/paper_registry.csv': 'bd8966638957fd2bf838d669405b90c77480207d8bf05154b13678eea869f8c6'}


def test_paper_configs_and_registry_remain_unchanged():
    actual_configs = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "configs/paper").rglob("*") if path.is_file()
    }
    assert actual_configs == {name for name in PAPER_SHA256 if name.startswith("configs/paper/")}
    for name, expected_hash in PAPER_SHA256.items():
        text = (ROOT / name).read_text(encoding="utf-8")
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == expected_hash, name

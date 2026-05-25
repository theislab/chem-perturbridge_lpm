#!/usr/bin/env python3
"""Create paper10 configs for Morgan fingerprint molecule-embedding variants."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "perturb_gym" / "configs" / "collection"
RESULTS_DIR = REPO_ROOT / "results"
MORGAN_EMBEDDINGS_PATH = (
    REPO_ROOT
    / ".plib_cache"
    / "morgan_perturbation_embeddings"
    / "pubchem_morgan_radius2_nbits128"
    / "compound_embeddings.npy"
)

SEEDS = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
SOURCE_BASE_CONFIG_ID = (
    "lpm_multiout_transfer_source_all_data_plus_tahoe_novartis_op3_"
    "molholdout_h100_2x_bs4096_200epoch_lustre"
)
SINGLE_DATASET_BASE_CONFIG_ID = (
    "lpm_multiout_all_data_plus_tahoe_novartis_op3_"
    "molholdout_h100_2x_bs4096_200epoch_lustre"
)

DATASETS: list[tuple[str, str, tuple[str, ...]]] = [
    ("cigs_mce", "CIGS MCE", ("cigs_mce_",)),
    ("cigs_tcm", "CIGS TCM", ("cigs_tcm_",)),
    ("dilimap_train", "dilimap_train", ("dili_train_",)),
    ("gdpx2", "Ginkgo GDPx2", ("gdpx2_",)),
    ("lincs_phase1", "LINCS_phase1_level3_epsilon", ("l1000_phase1_",)),
    ("lincs_phase2", "LINCS_phase2_level3", ("l1000_phase2_",)),
    ("novartis", "Novartis MoABox DRUG-seq", ("novartis_",)),
    ("op3", "op3", ("op3_",)),
    ("sciplex", "srivatsan20_sciplex3", ("sciplex_",)),
    ("tahoe100", "tahoe100", ("tahoe_",)),
    ("vcpi_0001", "Ginkgo VCPI vcpi-0001 (tvc-bhr-009)", ("vcpi_0001_",)),
    ("vcpi_0002", "Ginkgo VCPI vcpi-0002 (tvc-kdl-010)", ("vcpi_0002_",)),
]

VARIANTS: dict[str, dict[str, Any]] = {
    "morgan_fixed": {
        "group": "lpm_paper10_morgan_fixed_molholdout_h100_bs4096_200epoch_lustre",
        "source_family": "all_datasets_morgan_fixed",
        "scratch_family": "scratch_target_only_morgan_fixed",
        "source_prefix": "lpm_paper10_morgan_fixed_all_data_plus_tahoe_novartis_op3",
        "scratch_prefix": "lpm_paper10_morgan_fixed_scratch",
        "freeze_source_morgan": True,
        "include_scratch": True,
        "source_stage": "source_all_data_frozen_morgan",
        "scratch_stage": "scratch_target_only_frozen_morgan",
        "tags": ["frozen-morgan-fingerprints"],
    },
    "morgan_learned": {
        "group": "lpm_paper10_morgan_init_learned_molholdout_h100_bs4096_200epoch_lustre",
        "source_family": "all_datasets_morgan_learned",
        "scratch_family": "",
        "source_prefix": "lpm_paper10_morgan_init_learned_all_data_plus_tahoe_novartis_op3",
        "scratch_prefix": "",
        "freeze_source_morgan": False,
        "include_scratch": False,
        "source_stage": "source_all_data_morgan_initialized_learned_embeddings",
        "scratch_stage": "",
        "tags": ["morgan-initialized", "learned-molecule-embeddings-on-all"],
    },
}


def gpu_count_for_sources(sources: list[str]) -> int:
    return 1 if len(sources) == 1 else 2


def matching_sources(all_sources: list[str], prefixes: tuple[str, ...]) -> list[str]:
    sources = [source for source in all_sources if source.startswith(prefixes)]
    if not sources:
        raise RuntimeError(f"No sources matched prefixes {prefixes}")
    return sources


def write_config(config_id: str, config: dict[str, Any]) -> None:
    (CONFIG_DIR / f"{config_id}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def set_seed(config: dict[str, Any], seed: int) -> None:
    config["environment_configs"] = [{"seed": seed}]


def set_trainer_gpus(config: dict[str, Any], gpu_count: int) -> None:
    trainer_pars = config["model_configs"][0]["model_args"]["lightning_trainer_pars"]
    trainer_pars["devices"] = gpu_count
    trainer_pars["strategy"] = "auto" if gpu_count == 1 else "ddp"


def set_common_model_options(config: dict[str, Any], *, source_run: bool) -> None:
    model_config = config["model_configs"][0]
    model_args = model_config["model_args"]
    model_config["save_model_after_training"] = False
    model_args["keep_best_validation_checkpoint"] = True
    model_args["epoch_checkpoint_every_n"] = 1 if source_run else 0
    model_args["epoch_checkpoint_save_last"] = False


def set_wandb(config: dict[str, Any], config_id: str, group: str, tags: list[str], extra_config: dict[str, Any]) -> None:
    wandb_config = config["model_configs"][0]["wandb_config"]
    wandb_config["name"] = config_id
    wandb_config["group"] = group
    wandb_config["tags"] = tags
    wandb_config.setdefault("config", {})
    wandb_config["config"].update(extra_config)


def set_morgan_embeddings(config: dict[str, Any], *, freeze: bool) -> None:
    model_args = config["model_configs"][0]["model_args"]
    model_args.pop("initialize_from_checkpoint", None)
    model_args.pop("freeze_initialized_perturbation_embeddings", None)
    model_args["resume_from_checkpoint"] = None
    model_args["pretrained_perturbation_embeddings_path"] = str(MORGAN_EMBEDDINGS_PATH)
    model_args["freeze_pretrained_perturbation_embeddings"] = freeze


def build_variant(variant_name: str) -> Path:
    if not MORGAN_EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError(f"Missing Morgan embeddings: {MORGAN_EMBEDDINGS_PATH}")

    variant = VARIANTS[variant_name]
    source_base = yaml.safe_load((CONFIG_DIR / f"{SOURCE_BASE_CONFIG_ID}.yaml").read_text())
    single_base = yaml.safe_load((CONFIG_DIR / f"{SINGLE_DATASET_BASE_CONFIG_ID}.yaml").read_text())
    all_sources = list(single_base["data_configs"][0]["on_disk_data_sources"])

    rows: list[dict[str, str]] = []
    for seed in SEEDS:
        source_config_id = (
            f"{variant['source_prefix']}_seed{seed}_molholdout_h100_2x_bs4096_200epoch_lustre"
        )
        source = deepcopy(source_base)
        set_seed(source, seed)
        set_trainer_gpus(source, 2)
        set_common_model_options(source, source_run=True)
        set_morgan_embeddings(source, freeze=bool(variant["freeze_source_morgan"]))
        set_wandb(
            source,
            source_config_id,
            variant["group"],
            [
                "lpm",
                "multiout",
                "all-data",
                "paper10",
                "molecule-holdout",
                "source-all-data",
                *variant["tags"],
                f"seed-{seed}",
                "h100-2x",
                "bs4096",
            ],
            {
                "paper10_stage": variant["source_stage"],
                "seed": seed,
                "morgan_radius": 2,
                "morgan_nbits": 128,
                "pretrained_perturbation_embeddings_path": str(MORGAN_EMBEDDINGS_PATH),
                "freeze_pretrained_perturbation_embeddings": bool(variant["freeze_source_morgan"]),
                "source_selection": "per_dataset_best_validation_across_source_seeds_and_epochs",
            },
        )
        write_config(source_config_id, source)
        rows.append(
            {
                "model_family": variant["source_family"],
                "seed": str(seed),
                "dataset_slug": "all",
                "dataset": "all",
                "config_id": source_config_id,
                "gpus": "2",
                "source_count": str(len(all_sources)),
                "morgan_variant": variant_name,
                "pretrained_perturbation_embeddings_path": str(MORGAN_EMBEDDINGS_PATH),
                "freeze_pretrained_perturbation_embeddings": str(bool(variant["freeze_source_morgan"])),
            }
        )

        if not variant["include_scratch"]:
            continue
        for slug, dataset_name, prefixes in DATASETS:
            sources = matching_sources(all_sources, prefixes)
            gpu_count = gpu_count_for_sources(sources)
            scratch_config_id = (
                f"{variant['scratch_prefix']}_{slug}_seed{seed}_"
                f"molholdout_h100_{gpu_count}x_bs4096_200epoch_lustre"
            )
            scratch = deepcopy(single_base)
            set_seed(scratch, seed)
            data_config = scratch["data_configs"][0]
            data_config["on_disk_data_sources"] = sources
            if slug == "vcpi_0002":
                data_config.pop("val_and_test_perturbations_selected_from", None)
                data_config["val_perturbations_selected_from"] = "all"
            set_trainer_gpus(scratch, gpu_count)
            set_common_model_options(scratch, source_run=False)
            set_morgan_embeddings(scratch, freeze=True)
            set_wandb(
                scratch,
                scratch_config_id,
                variant["group"],
                [
                    "lpm",
                    "multiout",
                    "paper10",
                    "single-dataset",
                    "scratch",
                    "frozen-morgan-fingerprints",
                    "molecule-holdout",
                    slug,
                    f"seed-{seed}",
                    f"h100-{gpu_count}x",
                    "bs4096",
                ],
                {
                    "paper10_stage": variant["scratch_stage"],
                    "single_dataset": dataset_name,
                    "single_dataset_slug": slug,
                    "seed": seed,
                    "morgan_radius": 2,
                    "morgan_nbits": 128,
                    "pretrained_perturbation_embeddings_path": str(MORGAN_EMBEDDINGS_PATH),
                    "freeze_pretrained_perturbation_embeddings": True,
                },
            )
            write_config(scratch_config_id, scratch)
            rows.append(
                {
                    "model_family": variant["scratch_family"],
                    "seed": str(seed),
                    "dataset_slug": slug,
                    "dataset": dataset_name,
                    "config_id": scratch_config_id,
                    "gpus": str(gpu_count),
                    "source_count": str(len(sources)),
                    "morgan_variant": variant_name,
                    "pretrained_perturbation_embeddings_path": str(MORGAN_EMBEDDINGS_PATH),
                    "freeze_pretrained_perturbation_embeddings": "True",
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS_DIR / f"lpm_paper10_{variant_name}_config_manifest.tsv"
    with manifest_path.open("w", newline="") as handle:
        fieldnames = [
            "model_family",
            "seed",
            "dataset_slug",
            "dataset",
            "config_id",
            "gpus",
            "source_count",
            "morgan_variant",
            "pretrained_perturbation_embeddings_path",
            "freeze_pretrained_perturbation_embeddings",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    args = parser.parse_args()

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    for variant_name in variants:
        print(build_variant(variant_name))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create 10-seed LPM paper-table source and scratch configs.

Fine-tune configs are generated later by select_and_submit_lpm_paper10_finetune_jobs.py,
after the 10 all-dataset source runs have produced per-epoch checkpoints and validation
curves. That selector chooses the source checkpoint with best validation RMSE for each
dataset before creating the fine-tune configs.
"""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "perturb_gym" / "configs" / "collection"
RESULTS_DIR = REPO_ROOT / "results"

SEEDS = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
SOURCE_BASE_CONFIG_ID = (
    "lpm_multiout_transfer_source_all_data_plus_tahoe_novartis_op3_"
    "molholdout_h100_2x_bs4096_200epoch_lustre"
)
SINGLE_DATASET_BASE_CONFIG_ID = (
    "lpm_multiout_all_data_plus_tahoe_novartis_op3_"
    "molholdout_h100_2x_bs4096_200epoch_lustre"
)
GROUP = "lpm_paper10_molholdout_h100_bs4096_200epoch_lustre"

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


def gpu_count_for_sources(sources: list[str]) -> int:
    return 1 if len(sources) == 1 else 2


def write_config(config_id: str, config: dict) -> None:
    (CONFIG_DIR / f"{config_id}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def set_seed(config: dict, seed: int) -> None:
    config["environment_configs"] = [{"seed": seed}]


def set_trainer_gpus(config: dict, gpu_count: int) -> None:
    trainer_pars = config["model_configs"][0]["model_args"]["lightning_trainer_pars"]
    trainer_pars["devices"] = gpu_count
    trainer_pars["strategy"] = "auto" if gpu_count == 1 else "ddp"


def set_common_model_options(config: dict, *, source_run: bool) -> None:
    model_config = config["model_configs"][0]
    model_args = model_config["model_args"]
    model_config["save_model_after_training"] = False
    model_args["keep_best_validation_checkpoint"] = True
    if source_run:
        # Source checkpoints are later selected per dataset, so keep every epoch.
        model_args["epoch_checkpoint_every_n"] = 1
        model_args["epoch_checkpoint_save_last"] = False
    else:
        # For target-only/fine-tune runs the best-validation checkpoint is enough.
        model_args["epoch_checkpoint_every_n"] = 0
        model_args["epoch_checkpoint_save_last"] = False


def set_wandb(config: dict, config_id: str, tags: list[str], extra_config: dict) -> None:
    wandb_config = config["model_configs"][0]["wandb_config"]
    wandb_config["name"] = config_id
    wandb_config["group"] = GROUP
    wandb_config["tags"] = tags
    wandb_config.setdefault("config", {})
    wandb_config["config"].update(extra_config)


def matching_sources(all_sources: list[str], prefixes: tuple[str, ...]) -> list[str]:
    sources = [source for source in all_sources if source.startswith(prefixes)]
    if not sources:
        raise RuntimeError(f"No sources matched prefixes {prefixes}")
    return sources


def main() -> None:
    source_base = yaml.safe_load((CONFIG_DIR / f"{SOURCE_BASE_CONFIG_ID}.yaml").read_text())
    single_base = yaml.safe_load((CONFIG_DIR / f"{SINGLE_DATASET_BASE_CONFIG_ID}.yaml").read_text())
    all_sources = list(single_base["data_configs"][0]["on_disk_data_sources"])

    rows: list[dict[str, str]] = []

    for seed in SEEDS:
        source_config_id = (
            "lpm_paper10_all_data_plus_tahoe_novartis_op3_"
            f"seed{seed}_molholdout_h100_2x_bs4096_200epoch_lustre"
        )
        config = deepcopy(source_base)
        set_seed(config, seed)
        set_trainer_gpus(config, 2)
        set_common_model_options(config, source_run=True)
        model_args = config["model_configs"][0]["model_args"]
        model_args.pop("pretrained_perturbation_embeddings_path", None)
        model_args.pop("freeze_pretrained_perturbation_embeddings", None)
        model_args.pop("initialize_from_checkpoint", None)
        model_args.pop("freeze_initialized_perturbation_embeddings", None)
        model_args["resume_from_checkpoint"] = None
        set_wandb(
            config,
            source_config_id,
            [
                "lpm",
                "multiout",
                "all-data",
                "paper10",
                "molecule-holdout",
                "source-all-data",
                f"seed-{seed}",
                "h100-2x",
                "bs4096",
            ],
            {
                "paper10_stage": "source_all_data",
                "seed": seed,
                "source_selection": "per_dataset_best_validation_across_source_seeds_and_epochs",
            },
        )
        write_config(source_config_id, config)
        rows.append(
            {
                "model_family": "all_datasets",
                "seed": str(seed),
                "dataset_slug": "all",
                "dataset": "all",
                "config_id": source_config_id,
                "gpus": "2",
                "source_count": str(len(all_sources)),
            }
        )

        for slug, dataset_name, prefixes in DATASETS:
            sources = matching_sources(all_sources, prefixes)
            gpu_count = gpu_count_for_sources(sources)
            scratch_config_id = (
                f"lpm_paper10_scratch_{slug}_seed{seed}_"
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
            model_args = scratch["model_configs"][0]["model_args"]
            model_args.pop("pretrained_perturbation_embeddings_path", None)
            model_args.pop("freeze_pretrained_perturbation_embeddings", None)
            model_args.pop("initialize_from_checkpoint", None)
            model_args.pop("freeze_initialized_perturbation_embeddings", None)
            model_args["resume_from_checkpoint"] = None
            set_wandb(
                scratch,
                scratch_config_id,
                [
                    "lpm",
                    "multiout",
                    "paper10",
                    "single-dataset",
                    "scratch",
                    "molecule-holdout",
                    slug,
                    f"seed-{seed}",
                    f"h100-{gpu_count}x",
                    "bs4096",
                ],
                {
                    "paper10_stage": "scratch_target_only",
                    "single_dataset": dataset_name,
                    "single_dataset_slug": slug,
                    "seed": seed,
                },
            )
            write_config(scratch_config_id, scratch)
            rows.append(
                {
                    "model_family": "scratch_target_only",
                    "seed": str(seed),
                    "dataset_slug": slug,
                    "dataset": dataset_name,
                    "config_id": scratch_config_id,
                    "gpus": str(gpu_count),
                    "source_count": str(len(sources)),
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS_DIR / "lpm_paper10_config_manifest.tsv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model_family", "seed", "dataset_slug", "dataset", "config_id", "gpus", "source_count"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(manifest_path)


if __name__ == "__main__":
    main()

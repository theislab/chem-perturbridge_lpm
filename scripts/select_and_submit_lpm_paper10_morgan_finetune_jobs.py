#!/usr/bin/env python3
"""Select Morgan source checkpoints and submit target fine-tunes."""

from __future__ import annotations

import argparse
import csv
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "perturb_gym" / "configs" / "collection"
RESULTS_DIR = REPO_ROOT / "results"
PLIB_RESULTS_DIR = REPO_ROOT / ".plib_cache" / "results"
SINGLE_DATASET_BASE_CONFIG_ID = (
    "lpm_multiout_all_data_plus_tahoe_novartis_op3_"
    "molholdout_h100_2x_bs4096_200epoch_lustre"
)
SEEDS = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

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
        "finetune_family": "finetune_morgan_fixed",
        "finetune_prefix": "lpm_paper10_morgan_fixed_finetune_fixmol",
        "selection_prefix": "lpm_paper10_morgan_fixed_source_dataset_checkpoint_selection",
        "eval_prefix": "lpm_paper10_morgan_fixed_source_dataset_eval_seed",
        "job_prefix": "p10_mfix",
        "paper10_stage": "finetune_from_frozen_morgan_all_data",
        "tags": ["frozen-morgan-fingerprints", "fine-tune-from-morgan-all", "fixed-molecule-embeddings"],
    },
    "morgan_learned": {
        "group": "lpm_paper10_morgan_init_learned_molholdout_h100_bs4096_200epoch_lustre",
        "source_family": "all_datasets_morgan_learned",
        "scratch_family": "",
        "finetune_family": "finetune_morgan_learned_fixed_updated_embeddings",
        "finetune_prefix": "lpm_paper10_morgan_init_learned_finetune_fixmol",
        "selection_prefix": "lpm_paper10_morgan_learned_source_dataset_checkpoint_selection",
        "eval_prefix": "lpm_paper10_morgan_learned_source_dataset_eval_seed",
        "job_prefix": "p10_mlrn",
        "paper10_stage": "finetune_from_morgan_initialized_all_data_freeze_updated_embeddings",
        "tags": [
            "morgan-initialized",
            "source-molecule-embeddings-learned",
            "fine-tune-fixed-updated-molecule-embeddings",
        ],
    },
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def event_files(config_id: str, seed: int) -> list[Path]:
    return sorted(
        (PLIB_RESULTS_DIR / config_id).glob(f"LPM_*/seed_{seed}/learning_curves/version_*/events.out.tfevents*"),
        key=lambda path: path.stat().st_mtime,
    )


def scalar_values(config_id: str, seed: int, tag: str) -> list[float]:
    values: list[float] = []
    for path in event_files(config_id, seed):
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        values.extend(float(event.value) for event in accumulator.Scalars(tag))
    return values


def best_epoch_and_value(config_id: str, seed: int, dataset_name: str) -> tuple[int, float, str]:
    tag = f"Validation RMSE dataset/{dataset_name}"
    values = scalar_values(config_id, seed, tag)
    if not values:
        tag = "Validation RMSE"
        values = scalar_values(config_id, seed, tag)
    if not values:
        raise RuntimeError(f"Missing validation RMSE scalar for {config_id} seed {seed}")
    best_epoch, best_value = min(enumerate(values), key=lambda item: item[1])
    return int(best_epoch), float(best_value), tag


def checkpoint_for_epoch(config_id: str, seed: int, epoch: int) -> Path:
    checkpoint_dirs = sorted((PLIB_RESULTS_DIR / config_id).glob(f"LPM_*/seed_{seed}/checkpoints"))
    if not checkpoint_dirs:
        raise RuntimeError(f"No checkpoint directory found for {config_id} seed {seed}")
    epoch_text = f"{epoch:04d}"
    matches: list[Path] = []
    for checkpoint_dir in checkpoint_dirs:
        matches.extend(checkpoint_dir.glob(f"epoch-*{epoch_text}.ckpt"))
    if not matches:
        raise RuntimeError(f"No epoch checkpoint for {config_id} seed {seed} epoch {epoch}")
    return sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def gpu_count_for_sources(sources: list[str]) -> int:
    return 1 if len(sources) == 1 else 2


def matching_sources(all_sources: list[str], prefixes: tuple[str, ...]) -> list[str]:
    sources = [source for source in all_sources if source.startswith(prefixes)]
    if not sources:
        raise RuntimeError(f"No sources matched prefixes {prefixes}")
    return sources


def set_seed(config: dict[str, Any], seed: int) -> None:
    config["environment_configs"] = [{"seed": seed}]


def set_trainer_gpus(config: dict[str, Any], gpu_count: int) -> None:
    trainer_pars = config["model_configs"][0]["model_args"]["lightning_trainer_pars"]
    trainer_pars["devices"] = gpu_count
    trainer_pars["strategy"] = "auto" if gpu_count == 1 else "ddp"


def set_common_finetune_options(config: dict[str, Any]) -> None:
    model_config = config["model_configs"][0]
    model_args = model_config["model_args"]
    model_config["save_model_after_training"] = False
    model_args["keep_best_validation_checkpoint"] = True
    model_args["epoch_checkpoint_every_n"] = 0
    model_args["epoch_checkpoint_save_last"] = False


def set_wandb(config: dict[str, Any], config_id: str, group: str, tags: list[str], extra_config: dict[str, Any]) -> None:
    wandb_config = config["model_configs"][0]["wandb_config"]
    wandb_config["name"] = config_id
    wandb_config["group"] = group
    wandb_config["tags"] = tags
    wandb_config.setdefault("config", {})
    wandb_config["config"].update(extra_config)


def write_config(config_id: str, config: dict[str, Any]) -> None:
    (CONFIG_DIR / f"{config_id}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def submit_job(cmd: list[str], dry_run: bool = False) -> str:
    print(" ".join(cmd))
    if dry_run:
        return "DRYRUN"
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    job_id = completed.stdout.strip().split(";")[0]
    print(job_id)
    return job_id


def submit_training_job(config_id: str, gpus: int, job_name: str, dry_run: bool) -> str:
    return submit_job(
        [
            "sbatch",
            "--parsable",
            "--time=24:00:00",
            f"--gpus={gpus}",
            f"--job-name={job_name[:120]}",
            "--exclude=supergpu17,supergpu32",
            "--export=ALL,PREPARE_MULTI_OUTPUT_DATA=0,CLEAN_RESULTS=1,PERTURB_GYM_EXTRACT_COMPOUND_EMBEDDINGS=0",
            "run_single_node_lightning.sh",
            "--config",
            config_id,
        ],
        dry_run=dry_run,
    )


def submit_source_eval_job(source_seed: int, variant: dict[str, Any], selection_long_path: Path, dry_run: bool) -> str:
    output = RESULTS_DIR / f"{variant['eval_prefix']}{source_seed}.tsv"
    return submit_job(
        [
            "sbatch",
            "--parsable",
            "--time=06:00:00",
            "--gpus=1",
            f"--job-name={variant['job_prefix']}_eval_s{source_seed}",
            "--exclude=supergpu17,supergpu32",
            "run_lpm_paper10_source_eval.sh",
            "--source-seed",
            str(source_seed),
            "--selection-long",
            str(selection_long_path),
            "--model-family",
            str(variant["source_family"]),
            "--output",
            str(output),
        ],
        dry_run=dry_run,
    )


def submit_summary_job(dependency_job_ids: list[str], variant_name: str, dry_run: bool) -> str:
    dependency = "afterany:" + ":".join(job_id for job_id in dependency_job_ids if job_id != "DRYRUN")
    cmd = ["sbatch", "--parsable", f"--job-name=p10_{variant_name}_summary"]
    if dependency != "afterany:":
        cmd.append(f"--dependency={dependency}")
    cmd.append("run_lpm_paper10_summary.sh")
    return submit_job(cmd, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--config-manifest", type=Path, required=True)
    parser.add_argument("--source-scratch-jobs", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    manifest = read_tsv(args.config_manifest)
    source_rows = [row for row in manifest if row["model_family"] == variant["source_family"]]
    if len(source_rows) != len(SEEDS):
        raise RuntimeError(f"Expected {len(SEEDS)} source rows, found {len(source_rows)}")

    selection_rows: list[dict[str, str]] = []
    for source_row in source_rows:
        seed = int(source_row["seed"])
        config_id = source_row["config_id"]
        for slug, dataset_name, _ in DATASETS:
            epoch, value, selection_metric = best_epoch_and_value(config_id, seed, dataset_name)
            checkpoint_path = checkpoint_for_epoch(config_id, seed, epoch)
            selection_rows.append(
                {
                    "dataset_slug": slug,
                    "dataset": dataset_name,
                    "source_seed": str(seed),
                    "source_config_id": config_id,
                    "best_epoch": str(epoch),
                    "val_rmse": f"{value:.8f}",
                    "selection_metric": selection_metric,
                    "checkpoint_path": str(checkpoint_path),
                    "morgan_variant": args.variant,
                }
            )

    selection_long_path = RESULTS_DIR / f"{variant['selection_prefix']}_long.tsv"
    write_tsv(selection_long_path, selection_rows)
    best_by_dataset: list[dict[str, str]] = []
    for slug, _, _ in DATASETS:
        candidates = [row for row in selection_rows if row["dataset_slug"] == slug]
        best_by_dataset.append(min(candidates, key=lambda row: float(row["val_rmse"])))
    selection_path = RESULTS_DIR / f"{variant['selection_prefix']}.tsv"
    write_tsv(selection_path, best_by_dataset)

    base = yaml.safe_load((CONFIG_DIR / f"{SINGLE_DATASET_BASE_CONFIG_ID}.yaml").read_text())
    all_sources = list(base["data_configs"][0]["on_disk_data_sources"])
    selected_by_slug = {row["dataset_slug"]: row for row in best_by_dataset}

    finetune_rows: list[dict[str, str]] = []
    job_rows: list[dict[str, str]] = []
    for seed in SEEDS:
        for slug, dataset_name, prefixes in DATASETS:
            sources = matching_sources(all_sources, prefixes)
            gpu_count = gpu_count_for_sources(sources)
            selected = selected_by_slug[slug]
            config_id = (
                f"{variant['finetune_prefix']}_{slug}_seed{seed}_"
                f"molholdout_h100_{gpu_count}x_bs4096_200epoch_lustre"
            )
            config = deepcopy(base)
            set_seed(config, seed)
            data_config = config["data_configs"][0]
            data_config["on_disk_data_sources"] = sources
            if slug == "vcpi_0002":
                data_config.pop("val_and_test_perturbations_selected_from", None)
                data_config["val_perturbations_selected_from"] = "all"
            set_trainer_gpus(config, gpu_count)
            set_common_finetune_options(config)
            model_args = config["model_configs"][0]["model_args"]
            model_args.pop("pretrained_perturbation_embeddings_path", None)
            model_args.pop("freeze_pretrained_perturbation_embeddings", None)
            model_args["resume_from_checkpoint"] = None
            model_args["initialize_from_checkpoint"] = selected["checkpoint_path"]
            model_args["freeze_initialized_perturbation_embeddings"] = True
            set_wandb(
                config,
                config_id,
                variant["group"],
                [
                    "lpm",
                    "multiout",
                    "paper10",
                    "single-dataset",
                    "fine-tune-from-all",
                    *variant["tags"],
                    "molecule-holdout",
                    slug,
                    f"seed-{seed}",
                    f"h100-{gpu_count}x",
                    "bs4096",
                ],
                {
                    "paper10_stage": variant["paper10_stage"],
                    "single_dataset": dataset_name,
                    "single_dataset_slug": slug,
                    "seed": seed,
                    "initialized_from_source_seed": int(selected["source_seed"]),
                    "initialized_from_source_config": selected["source_config_id"],
                    "initialize_from_checkpoint": selected["checkpoint_path"],
                    "source_checkpoint_dataset_val_rmse": float(selected["val_rmse"]),
                    "source_checkpoint_dataset_best_epoch": int(selected["best_epoch"]),
                    "source_checkpoint_selection_metric": selected["selection_metric"],
                    "freeze_initialized_perturbation_embeddings": True,
                    "morgan_variant": args.variant,
                },
            )
            write_config(config_id, config)
            finetune_rows.append(
                {
                    "model_family": variant["finetune_family"],
                    "seed": str(seed),
                    "dataset_slug": slug,
                    "dataset": dataset_name,
                    "config_id": config_id,
                    "gpus": str(gpu_count),
                    "source_count": str(len(sources)),
                    "initialize_from_checkpoint": selected["checkpoint_path"],
                    "initialized_from_source_seed": selected["source_seed"],
                    "source_checkpoint_dataset_val_rmse": selected["val_rmse"],
                    "source_checkpoint_selection_metric": selected["selection_metric"],
                    "morgan_variant": args.variant,
                    "freeze_initialized_perturbation_embeddings": "True",
                }
            )
            job_id = submit_training_job(config_id, gpu_count, f"{variant['job_prefix']}_ft_{slug}_s{seed}", args.dry_run)
            job_rows.append({**finetune_rows[-1], "job_id": job_id})

    finetune_manifest = RESULTS_DIR / f"lpm_paper10_{args.variant}_finetune_config_manifest.tsv"
    write_tsv(finetune_manifest, finetune_rows)
    finetune_jobs = RESULTS_DIR / f"lpm_paper10_{args.variant}_finetune_slurm_jobs.tsv"
    write_tsv(finetune_jobs, job_rows)

    eval_job_rows = []
    for seed in SEEDS:
        job_id = submit_source_eval_job(seed, variant, selection_long_path, args.dry_run)
        eval_job_rows.append({"source_seed": str(seed), "job_id": job_id, "morgan_variant": args.variant})
    eval_jobs = RESULTS_DIR / f"lpm_paper10_{args.variant}_source_eval_slurm_jobs.tsv"
    write_tsv(eval_jobs, eval_job_rows)

    upstream_jobs = read_tsv(args.source_scratch_jobs)
    scratch_job_ids = [
        row["job_id"]
        for row in upstream_jobs
        if row.get("model_family") == variant["scratch_family"] and row.get("job_id")
    ]
    summary_job_id = submit_summary_job(
        [*scratch_job_ids, *[row["job_id"] for row in job_rows], *[row["job_id"] for row in eval_job_rows]],
        args.variant,
        args.dry_run,
    )
    summary_jobs = RESULTS_DIR / f"lpm_paper10_{args.variant}_summary_slurm_job.tsv"
    write_tsv(summary_jobs, [{"job_id": summary_job_id}], fieldnames=["job_id"])

    print(f"Wrote {selection_long_path}")
    print(f"Wrote {selection_path}")
    print(f"Wrote {finetune_manifest}")
    print(f"Wrote {finetune_jobs}")
    print(f"Wrote {eval_jobs}")
    print(f"Wrote {summary_jobs}")


if __name__ == "__main__":
    main()

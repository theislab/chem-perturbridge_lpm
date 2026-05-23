#!/usr/bin/env python3
"""Create LPM multi-output hyperparameter sweep configs.

The sweep is intentionally explicit rather than random at submission time so
that the Slurm jobs, W&B runs, and final summary all refer to stable run ids.
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "perturb_gym" / "configs" / "collection"
DEFAULT_BASE_CONFIG = (
    CONFIG_DIR / "lpm_multiout_all_data_plus_tahoe_novartis_op3_molholdout_h100_2x_bs4096_200epoch_lustre.yaml"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "lpm_hparam_sweep_all_op3_molholdout"


BASELINE_CONFIG_ID = "lpm_multiout_all_data_plus_tahoe_novartis_op3_molholdout_h100_2x_bs4096_200epoch_lustre"


def candidate_specs(max_epochs: int) -> list[dict[str, Any]]:
    base = {
        "optimizer_name": "AdamW",
        "max_epochs": max_epochs,
    }
    specs = [
        {
            "run_name": "run00_baseline_existing",
            "config_id": BASELINE_CONFIG_ID,
            "submit": "existing",
            "job_id": "36693408",
            "learning_rate": 0.002,
            "learning_rate_decay": 0.97,
            "num_layers": 2,
            "hidden_dim": 256,
            "embedding_dim": 128,
            "dropout": 0.10,
            "batch_size": 4096,
            "num_workers": 4,
        },
        {
            "run_name": "run01_lr1e3_decay99_l2_h256_e128_d10_bs4096",
            "learning_rate": 0.001,
            "learning_rate_decay": 0.99,
            "num_layers": 2,
            "hidden_dim": 256,
            "embedding_dim": 128,
            "dropout": 0.10,
            "batch_size": 4096,
            "num_workers": 4,
        },
        {
            "run_name": "run02_lr3e3_decay97_l2_h256_e128_d10_bs4096",
            "learning_rate": 0.003,
            "learning_rate_decay": 0.97,
            "num_layers": 2,
            "hidden_dim": 256,
            "embedding_dim": 128,
            "dropout": 0.10,
            "batch_size": 4096,
            "num_workers": 4,
        },
        {
            "run_name": "run03_lr15e4_decay985_l2_h512_e128_d10_bs4096",
            "learning_rate": 0.0015,
            "learning_rate_decay": 0.985,
            "num_layers": 2,
            "hidden_dim": 512,
            "embedding_dim": 128,
            "dropout": 0.10,
            "batch_size": 4096,
            "num_workers": 4,
        },
        {
            "run_name": "run04_lr2e3_decay985_l3_h512_e128_d10_bs4096",
            "learning_rate": 0.002,
            "learning_rate_decay": 0.985,
            "num_layers": 3,
            "hidden_dim": 512,
            "embedding_dim": 128,
            "dropout": 0.10,
            "batch_size": 4096,
            "num_workers": 4,
        },
        {
            "run_name": "run05_lr1e3_decay99_l3_h512_e192_d10_bs4096",
            "learning_rate": 0.001,
            "learning_rate_decay": 0.99,
            "num_layers": 3,
            "hidden_dim": 512,
            "embedding_dim": 192,
            "dropout": 0.10,
            "batch_size": 4096,
            "num_workers": 4,
        },
        {
            "run_name": "run06_lr2e3_decay99_l1_h512_e128_d05_bs4096",
            "learning_rate": 0.002,
            "learning_rate_decay": 0.99,
            "num_layers": 1,
            "hidden_dim": 512,
            "embedding_dim": 128,
            "dropout": 0.05,
            "batch_size": 4096,
            "num_workers": 4,
        },
        {
            "run_name": "run07_lr2e3_decay97_l4_h512_e128_d20_bs4096",
            "learning_rate": 0.002,
            "learning_rate_decay": 0.97,
            "num_layers": 4,
            "hidden_dim": 512,
            "embedding_dim": 128,
            "dropout": 0.20,
            "batch_size": 4096,
            "num_workers": 4,
        },
        {
            "run_name": "run08_lr8e4_decay995_l3_h768_e128_d10_bs4096",
            "learning_rate": 0.0008,
            "learning_rate_decay": 0.995,
            "num_layers": 3,
            "hidden_dim": 768,
            "embedding_dim": 128,
            "dropout": 0.10,
            "batch_size": 4096,
            "num_workers": 4,
        },
        {
            "run_name": "run09_lr15e4_decay99_l2_h768_e192_d10_bs4096",
            "learning_rate": 0.0015,
            "learning_rate_decay": 0.99,
            "num_layers": 2,
            "hidden_dim": 768,
            "embedding_dim": 192,
            "dropout": 0.10,
            "batch_size": 4096,
            "num_workers": 4,
        },
        {
            "run_name": "run10_lr25e4_decay985_l2_h512_e256_d10_bs4096",
            "learning_rate": 0.0025,
            "learning_rate_decay": 0.985,
            "num_layers": 2,
            "hidden_dim": 512,
            "embedding_dim": 256,
            "dropout": 0.10,
            "batch_size": 4096,
            "num_workers": 4,
        },
        {
            "run_name": "run11_lr1e3_decay1_l2_h512_e256_d20_bs4096",
            "learning_rate": 0.001,
            "learning_rate_decay": 1.0,
            "num_layers": 2,
            "hidden_dim": 512,
            "embedding_dim": 256,
            "dropout": 0.20,
            "batch_size": 4096,
            "num_workers": 4,
        },
        {
            "run_name": "run12_lr3e3_decay99_l3_h256_e128_d05_bs4096",
            "learning_rate": 0.003,
            "learning_rate_decay": 0.99,
            "num_layers": 3,
            "hidden_dim": 256,
            "embedding_dim": 128,
            "dropout": 0.05,
            "batch_size": 4096,
            "num_workers": 4,
        },
        {
            "run_name": "run13_lr2e3_decay985_l2_h384_e128_d0_bs8192",
            "learning_rate": 0.002,
            "learning_rate_decay": 0.985,
            "num_layers": 2,
            "hidden_dim": 384,
            "embedding_dim": 128,
            "dropout": 0.0,
            "batch_size": 8192,
            "num_workers": 8,
        },
        {
            "run_name": "run14_lr15e4_decay99_l3_h384_e128_d10_bs8192",
            "learning_rate": 0.0015,
            "learning_rate_decay": 0.99,
            "num_layers": 3,
            "hidden_dim": 384,
            "embedding_dim": 128,
            "dropout": 0.10,
            "batch_size": 8192,
            "num_workers": 8,
        },
        {
            "run_name": "run15_lr3e3_decay97_l2_h384_e192_d20_bs8192",
            "learning_rate": 0.003,
            "learning_rate_decay": 0.97,
            "num_layers": 2,
            "hidden_dim": 384,
            "embedding_dim": 192,
            "dropout": 0.20,
            "batch_size": 8192,
            "num_workers": 8,
        },
        {
            "run_name": "run16_lr1e3_decay995_l4_h384_e192_d15_bs4096",
            "learning_rate": 0.001,
            "learning_rate_decay": 0.995,
            "num_layers": 4,
            "hidden_dim": 384,
            "embedding_dim": 192,
            "dropout": 0.15,
            "batch_size": 4096,
            "num_workers": 4,
        },
        {
            "run_name": "run17_lr5e4_decay1_l2_h512_e128_d10_bs2048",
            "learning_rate": 0.0005,
            "learning_rate_decay": 1.0,
            "num_layers": 2,
            "hidden_dim": 512,
            "embedding_dim": 128,
            "dropout": 0.10,
            "batch_size": 2048,
            "num_workers": 4,
        },
        {
            "run_name": "run18_lr2e3_decay99_l1_h256_e128_d0_bs2048",
            "learning_rate": 0.002,
            "learning_rate_decay": 0.99,
            "num_layers": 1,
            "hidden_dim": 256,
            "embedding_dim": 128,
            "dropout": 0.0,
            "batch_size": 2048,
            "num_workers": 4,
        },
        {
            "run_name": "run19_lr4e3_decay97_l2_h256_e128_d20_bs8192",
            "learning_rate": 0.004,
            "learning_rate_decay": 0.97,
            "num_layers": 2,
            "hidden_dim": 256,
            "embedding_dim": 128,
            "dropout": 0.20,
            "batch_size": 8192,
            "num_workers": 8,
        },
    ]
    for idx, spec in enumerate(specs):
        spec.update(base)
        if "config_id" not in spec:
            spec["config_id"] = f"lpm_multiout_all_op3_molholdout_hparam_{idx:02d}"
        if "submit" not in spec:
            spec["submit"] = "array"
        if "job_id" not in spec:
            spec["job_id"] = ""
    return specs


def update_config(base_config: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    model_config = config["model_configs"][0]
    model_args = model_config["model_args"]
    trainer_args = model_args["lightning_trainer_pars"]

    for key in (
        "optimizer_name",
        "learning_rate",
        "learning_rate_decay",
        "num_layers",
        "hidden_dim",
        "embedding_dim",
        "dropout",
        "batch_size",
        "num_workers",
    ):
        model_args[key] = spec[key]

    model_args["epoch_checkpoint_every_n"] = 0
    model_args["epoch_checkpoint_save_last"] = False
    model_args["keep_best_validation_checkpoint"] = True
    model_config["save_model_after_training"] = False
    trainer_args["max_epochs"] = spec["max_epochs"]
    trainer_args["log_every_n_steps"] = 1

    wandb_config = model_config.setdefault("wandb_config", {})
    wandb_config["name"] = spec["config_id"]
    wandb_config["group"] = "lpm_hparam_sweep_all_op3_molholdout"
    tags = list(wandb_config.get("tags", []))
    for tag in ("hparam-sweep", "all-data-op3", "molecule-holdout"):
        if tag not in tags:
            tags.append(tag)
    wandb_config["tags"] = tags
    wandb_config["config"] = {
        "sweep_name": "lpm_hparam_sweep_all_op3_molholdout",
        "run_name": spec["run_name"],
        "search_budget_runs": 20,
    }
    return config


def write_manifest(path: Path, specs: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_name",
        "config_id",
        "submit",
        "job_id",
        "learning_rate",
        "learning_rate_decay",
        "num_layers",
        "hidden_dim",
        "embedding_dim",
        "dropout",
        "batch_size",
        "num_workers",
        "max_epochs",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for spec in specs:
            writer.writerow(spec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-epochs", type=int, default=200)
    args = parser.parse_args()

    with args.base_config.open("r") as handle:
        base_config = yaml.safe_load(handle)

    specs = candidate_specs(args.max_epochs)
    args.config_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        if spec["submit"] != "array":
            continue
        config = update_config(base_config, spec)
        config_path = args.config_dir / f"{spec['config_id']}.yaml"
        with config_path.open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    manifest_path = args.output_dir / "manifest.tsv"
    write_manifest(manifest_path, specs)
    print(f"wrote {sum(1 for spec in specs if spec['submit'] == 'array')} configs")
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()

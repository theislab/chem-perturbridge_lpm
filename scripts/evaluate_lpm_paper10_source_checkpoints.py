#!/usr/bin/env python3
"""Evaluate all-dataset source checkpoints selected per dataset for paper10."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import re
from pathlib import Path
from typing import Any

import polars as pl

import perturb_lib as plib
from perturb_gym.configs.access import load_training_configs
from perturb_lib.models.collection.lpm import LPM


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
SELECTION_LONG = RESULTS_DIR / "lpm_paper10_source_dataset_checkpoint_selection_long.tsv"

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
DATASET_PREFIXES = {slug: prefixes for slug, _, prefixes in DATASETS}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def matching_sources(all_sources: list[str], prefixes: tuple[str, ...]) -> list[str]:
    sources = [source for source in all_sources if source.startswith(prefixes)]
    if not sources:
        raise RuntimeError(f"No sources matched prefixes {prefixes}")
    return sources


def filter_to_sources(data: Any, sources: list[str]) -> Any | None:
    if data is None:
        return None
    if not hasattr(data, "_data") or "shard_path" not in data._data.columns:
        return data
    pattern = "|".join(re.escape(f"/{source}/") for source in sources)
    filtered_metadata = data._data.filter(pl.col("shard_path").str.contains(pattern))
    if filtered_metadata.is_empty():
        return None
    filtered = copy.copy(data)
    filtered._data = filtered_metadata
    if hasattr(filtered, "_shard_cache"):
        filtered._shard_cache = {}
    return filtered


def is_multiout_data(data: Any) -> bool:
    return {"dataset_code", "context_code", "perturbation_codes", "readout_codes"}.issubset(data.columns)


def predict_input(data: Any) -> Any:
    if is_multiout_data(data):
        return data.subset_columnwise(
            [
                "dataset_code",
                "context_code",
                "perturbation_codes",
                "log_dose",
                "time",
                "readout_codes",
                "n_values",
            ]
        )
    return data.subset_columnwise(["dataset", "context", "perturbation", "readout", "log_dose", "time"])


def rmse(model: LPM, data: Any | None) -> float | None:
    if data is None:
        return None
    try:
        if len(data) == 0:
            return None
    except Exception:
        pass
    evaluator = plib.load_evaluator("RMSE")
    predictions = model.predict(predict_input(data))
    value = float(evaluator.evaluate(predictions, data))
    return None if not math.isfinite(value) else value


def load_config_data(config_id: str):
    training_configs = list(load_training_configs(config_id))
    if len(training_configs) != 1:
        raise RuntimeError(f"Expected one training config for {config_id}, found {len(training_configs)}")
    return training_configs[0].get_train_val_test_data(), list(training_configs[0].data_config.on_disk_data_sources)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-seed", type=int, required=True)
    parser.add_argument("--selection-long", type=Path, default=SELECTION_LONG)
    parser.add_argument("--model-family", default="all_datasets")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = [row for row in read_tsv(args.selection_long) if int(row["source_seed"]) == args.source_seed]
    if not rows:
        raise RuntimeError(f"No selected source checkpoints found for seed {args.source_seed}")

    data_cache: dict[str, tuple[Any, Any | None, Any | None, list[str]]] = {}
    output_rows: list[dict[str, str]] = []
    for row in rows:
        config_id = row["source_config_id"]
        if config_id not in data_cache:
            (traindata, valdata, testdata), all_sources = load_config_data(config_id)
            data_cache[config_id] = (traindata, valdata, testdata, all_sources)
        _, valdata, testdata, all_sources = data_cache[config_id]

        slug = row["dataset_slug"]
        sources = matching_sources(all_sources, DATASET_PREFIXES[slug])
        dataset_val = filter_to_sources(valdata, sources)
        dataset_test = filter_to_sources(testdata, sources)

        checkpoint_path = Path(row["checkpoint_path"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        model = LPM.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
        model.num_workers = 0
        model.pin_memory = False
        model.eval()

        val_rmse = rmse(model, dataset_val)
        test_rmse = rmse(model, dataset_test)
        output_rows.append(
            {
                "model_family": args.model_family,
                "source_seed": row["source_seed"],
                "dataset_slug": slug,
                "dataset": row["dataset"],
                "source_config_id": config_id,
                "checkpoint_path": str(checkpoint_path),
                "selected_epoch": row["best_epoch"],
                "selected_val_rmse_from_tb": row["val_rmse"],
                "eval_val_rmse": "" if val_rmse is None else f"{val_rmse:.8f}",
                "eval_test_rmse": "" if test_rmse is None else f"{test_rmse:.8f}",
            }
        )

    output = args.output or RESULTS_DIR / f"lpm_paper10_source_dataset_eval_seed{args.source_seed}.tsv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)
    print(output)


if __name__ == "__main__":
    main()

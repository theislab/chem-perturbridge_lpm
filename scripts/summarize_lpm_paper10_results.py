#!/usr/bin/env python3
"""Summarize 10-seed paper-table LPM runs with mean +/- std validation/test RMSE."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PLIB_RESULTS_DIR = REPO_ROOT / ".plib_cache" / "results"

DATASETS: list[tuple[str, str]] = [
    ("cigs_mce", "CIGS MCE"),
    ("cigs_tcm", "CIGS TCM"),
    ("dilimap_train", "dilimap_train"),
    ("gdpx2", "Ginkgo GDPx2"),
    ("lincs_phase1", "LINCS_phase1_level3_epsilon"),
    ("lincs_phase2", "LINCS_phase2_level3"),
    ("novartis", "Novartis MoABox DRUG-seq"),
    ("op3", "op3"),
    ("sciplex", "srivatsan20_sciplex3"),
    ("tahoe100", "tahoe100"),
    ("vcpi_0001", "Ginkgo VCPI vcpi-0001 (tvc-bhr-009)"),
    ("vcpi_0002", "Ginkgo VCPI vcpi-0002 (tvc-kdl-010)"),
]
FAMILIES: list[tuple[str, str]] = [
    ("all_datasets", "all"),
    ("finetune_frozen_molecule_embeddings", "finetune_frozen_mol"),
    ("scratch_target_only", "scratch_target_only"),
    ("all_datasets_morgan_fixed", "all_morgan_fixed"),
    ("finetune_morgan_fixed", "finetune_morgan_fixed"),
    ("scratch_target_only_morgan_fixed", "scratch_morgan_fixed"),
    ("all_datasets_morgan_learned", "all_morgan_learned"),
    ("finetune_morgan_learned_fixed_updated_embeddings", "finetune_morgan_learned_fixmol"),
]
SOURCE_EVAL_PATTERNS: list[tuple[str, str]] = [
    ("lpm_paper10_source_dataset_eval_seed*.tsv", "all_datasets"),
    ("lpm_paper10_morgan_fixed_source_dataset_eval_seed*.tsv", "all_datasets_morgan_fixed"),
    ("lpm_paper10_morgan_learned_source_dataset_eval_seed*.tsv", "all_datasets_morgan_learned"),
]
CONFIG_MANIFESTS = [
    "lpm_paper10_config_manifest.tsv",
    "lpm_paper10_morgan_fixed_config_manifest.tsv",
    "lpm_paper10_morgan_learned_config_manifest.tsv",
]
FINETUNE_MANIFESTS = [
    "lpm_paper10_finetune_config_manifest.tsv",
    "lpm_paper10_morgan_fixed_finetune_config_manifest.tsv",
    "lpm_paper10_morgan_learned_finetune_config_manifest.tsv",
]
TARGET_RUN_FAMILIES = {
    "scratch_target_only",
    "finetune_frozen_molecule_embeddings",
    "scratch_target_only_morgan_fixed",
    "finetune_morgan_fixed",
    "finetune_morgan_learned_fixed_updated_embeddings",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def model_dir(config_id: str, seed: int) -> Path | None:
    root = PLIB_RESULTS_DIR / config_id
    candidates = sorted(root.glob(f"LPM_*/seed_{seed}"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def read_target_run_row(manifest_row: dict[str, str]) -> dict[str, str]:
    seed = int(manifest_row["seed"])
    config_id = manifest_row["config_id"]
    directory = model_dir(config_id, seed)
    result = {
        "model_family": manifest_row["model_family"],
        "seed": str(seed),
        "dataset_slug": manifest_row["dataset_slug"],
        "dataset": manifest_row["dataset"],
        "config_id": config_id,
        "checkpoint_path": "",
        "val_rmse": "",
        "test_rmse": "",
        "status": "missing_model_dir",
    }
    if directory is None:
        return result

    training_log_path = directory / "training_log.yaml"
    evaluation_log_path = directory / "evaluation_log.yaml"
    if training_log_path.exists():
        training_log = yaml.safe_load(training_log_path.read_text()) or {}
        result["checkpoint_path"] = str(training_log.get("best_validation_checkpoint") or "")
        val_from_log = parse_float(training_log.get("best_validation_rmse"))
        if val_from_log is not None:
            result["val_rmse"] = f"{val_from_log:.8f}"
    if evaluation_log_path.exists():
        evaluation_log = yaml.safe_load(evaluation_log_path.read_text()) or {}
        rmse = evaluation_log.get("RMSE", {}) or {}
        if not result["val_rmse"]:
            val_rmse = parse_float(rmse.get("val"))
            if val_rmse is not None:
                result["val_rmse"] = f"{val_rmse:.8f}"
        test_rmse = parse_float(rmse.get("test"))
        if test_rmse is not None:
            result["test_rmse"] = f"{test_rmse:.8f}"
        result["status"] = "ok"
    else:
        result["status"] = "missing_evaluation_log"
    return result


def load_long_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for pattern, default_family in SOURCE_EVAL_PATTERNS:
        for path in sorted(RESULTS_DIR.glob(pattern)):
            for row in read_tsv(path):
                family = row.get("model_family") or default_family
                rows.append(
                    {
                        "model_family": family,
                        "seed": row["source_seed"],
                        "dataset_slug": row["dataset_slug"],
                        "dataset": row["dataset"],
                        "config_id": row["source_config_id"],
                        "checkpoint_path": row["checkpoint_path"],
                        "val_rmse": row.get("eval_val_rmse", ""),
                        "test_rmse": row.get("eval_test_rmse", ""),
                        "status": "ok" if row.get("eval_val_rmse") else "missing_eval_val_rmse",
                    }
                )

    config_manifest: list[dict[str, str]] = []
    for manifest_name in CONFIG_MANIFESTS:
        config_manifest.extend(read_tsv(RESULTS_DIR / manifest_name))
    finetune_manifest: list[dict[str, str]] = []
    for manifest_name in FINETUNE_MANIFESTS:
        finetune_manifest.extend(read_tsv(RESULTS_DIR / manifest_name))
    for row in [*config_manifest, *finetune_manifest]:
        if row.get("model_family") not in TARGET_RUN_FAMILIES:
            continue
        rows.append(read_target_run_row(row))

    return rows


def summarize(values: list[float]) -> tuple[str, str, str]:
    if not values:
        return "", "", "0"
    avg = mean(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    return f"{avg:.8f}", f"{sd:.8f}", str(len(values))


def fmt_mean_std(mean_value: str, std_value: str, n_value: str) -> str:
    if not mean_value:
        return "NA"
    return f"{float(mean_value):.4f} +/- {float(std_value):.4f} (n={n_value})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", type=Path, default=RESULTS_DIR / "lpm_paper10_results")
    args = parser.parse_args()

    long_rows = load_long_rows()
    long_fieldnames = [
        "model_family",
        "seed",
        "dataset_slug",
        "dataset",
        "config_id",
        "checkpoint_path",
        "val_rmse",
        "test_rmse",
        "status",
    ]
    long_path = args.output_prefix.with_name(args.output_prefix.name + "_long.tsv")
    write_tsv(long_path, long_rows, long_fieldnames)

    summary_rows: list[dict[str, str]] = []
    for slug, dataset_name in DATASETS:
        row: dict[str, str] = {"dataset_slug": slug, "dataset": dataset_name}
        for family, prefix in FAMILIES:
            family_rows = [r for r in long_rows if r["dataset_slug"] == slug and r["model_family"] == family]
            val_values = [v for v in (parse_float(r["val_rmse"]) for r in family_rows) if v is not None]
            test_values = [v for v in (parse_float(r["test_rmse"]) for r in family_rows) if v is not None]
            val_mean, val_std, val_n = summarize(val_values)
            test_mean, test_std, test_n = summarize(test_values)
            row[f"{prefix}_val_rmse_mean"] = val_mean
            row[f"{prefix}_val_rmse_std"] = val_std
            row[f"{prefix}_val_n"] = val_n
            row[f"{prefix}_test_rmse_mean"] = test_mean
            row[f"{prefix}_test_rmse_std"] = test_std
            row[f"{prefix}_test_n"] = test_n
            row[f"{prefix}_val_rmse_mean_std"] = fmt_mean_std(val_mean, val_std, val_n)
            row[f"{prefix}_test_rmse_mean_std"] = fmt_mean_std(test_mean, test_std, test_n)
        summary_rows.append(row)

    summary_fieldnames = ["dataset_slug", "dataset"]
    for _, prefix in FAMILIES:
        summary_fieldnames.extend(
            [
                f"{prefix}_val_rmse_mean",
                f"{prefix}_val_rmse_std",
                f"{prefix}_val_n",
                f"{prefix}_test_rmse_mean",
                f"{prefix}_test_rmse_std",
                f"{prefix}_test_n",
                f"{prefix}_val_rmse_mean_std",
                f"{prefix}_test_rmse_mean_std",
            ]
        )
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.tsv")
    write_tsv(summary_path, summary_rows, summary_fieldnames)

    md_columns = ["dataset"]
    for _, prefix in FAMILIES:
        md_columns.extend([f"{prefix}_val_rmse_mean_std", f"{prefix}_test_rmse_mean_std"])
    lines = ["| " + " | ".join(md_columns) + " |", "| " + " | ".join(["---"] * len(md_columns)) + " |"]
    for row in summary_rows:
        lines.append("| " + " | ".join(row[col] for col in md_columns) + " |")
    md_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.md")
    md_path.write_text("\n".join(lines) + "\n")

    print(md_path.read_text())
    print(f"Wrote {long_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

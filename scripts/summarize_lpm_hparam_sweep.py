#!/usr/bin/env python3
"""Summarize the OP3-inclusive LPM hyperparameter sweep."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_DIR = REPO_ROOT / "results" / "lpm_hparam_sweep_all_op3_molholdout"
DEFAULT_RESULTS_ROOT = REPO_ROOT / ".plib_cache" / "results"
BEST_EPOCH_RE = re.compile(r"best-validation-rmse-epoch(\d+)\.ckpt$")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def latest_training_log(config_id: str, results_root: Path) -> Path | None:
    config_root = results_root / config_id
    logs = sorted(config_root.glob("LPM_*/seed_*/training_log.yaml"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def best_epoch_from_checkpoint(path_value: Any) -> int | None:
    if not path_value:
        return None
    match = BEST_EPOCH_RE.search(str(path_value))
    if not match:
        return None
    return int(match.group(1))


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def summarize_rows(manifest_rows: list[dict[str, str]], results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        config_id = manifest_row["config_id"]
        row: dict[str, Any] = dict(manifest_row)
        training_log = latest_training_log(config_id, results_root)
        row["training_log"] = "" if training_log is None else str(training_log)
        row["model_dir"] = "" if training_log is None else str(training_log.parent)
        row["status"] = "missing"
        row["best_validation_rmse"] = ""
        row["best_epoch"] = ""
        row["training_time"] = ""
        row["max_gpu_allocated"] = ""
        row["best_validation_checkpoint"] = ""
        if training_log is not None:
            with training_log.open("r") as handle:
                log = yaml.safe_load(handle) or {}
            row["status"] = "complete"
            row["best_validation_rmse"] = log.get("best_validation_rmse", "")
            row["best_validation_checkpoint"] = log.get("best_validation_checkpoint", "")
            row["best_epoch"] = best_epoch_from_checkpoint(row["best_validation_checkpoint"]) or ""
            row["training_time"] = log.get("training_time", "")
            row["max_gpu_allocated"] = log.get("max memory allocated per gpu", "")
        rows.append(row)

    completed = [row for row in rows if parse_float(row.get("best_validation_rmse")) is not None]
    completed.sort(key=lambda row: parse_float(row["best_validation_rmse"]) or float("inf"))
    rank_by_config = {row["config_id"]: idx + 1 for idx, row in enumerate(completed)}
    for row in rows:
        row["rank"] = rank_by_config.get(row["config_id"], "")
    rows.sort(key=lambda row: (row["rank"] == "", row["rank"] if row["rank"] != "" else 10**9, row["run_name"]))
    return rows


def write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "rank",
        "status",
        "run_name",
        "config_id",
        "submit",
        "job_id",
        "best_validation_rmse",
        "best_epoch",
        "learning_rate",
        "learning_rate_decay",
        "num_layers",
        "hidden_dim",
        "embedding_dim",
        "dropout",
        "batch_size",
        "num_workers",
        "max_epochs",
        "training_time",
        "max_gpu_allocated",
        "best_validation_checkpoint",
        "training_log",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def factor_summary(rows: list[dict[str, Any]], key: str) -> list[tuple[str, int, float, float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        metric = parse_float(row.get("best_validation_rmse"))
        if metric is None:
            continue
        groups[str(row.get(key, ""))].append(metric)
    summary = []
    for value, metrics in groups.items():
        summary.append((value, len(metrics), min(metrics), median(metrics)))
    summary.sort(key=lambda item: item[2])
    return summary


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    completed = [row for row in rows if parse_float(row.get("best_validation_rmse")) is not None]
    missing = [row for row in rows if parse_float(row.get("best_validation_rmse")) is None]
    lines: list[str] = []
    lines.append("# LPM Hyperparameter Sweep: All Data + OP3 Mol-Holdout")
    lines.append("")
    lines.append(f"Completed runs: {len(completed)} / {len(rows)}")
    if missing:
        lines.append(f"Incomplete or missing runs: {len(missing)}")
    lines.append("")
    if completed:
        lines.append("## Top Runs")
        lines.append("")
        lines.append(
            "| rank | run | val RMSE | best epoch | lr | decay | layers | hidden | emb | dropout | batch | workers |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in completed[:10]:
            lines.append(
                "| {rank} | {run_name} | {best_validation_rmse} | {best_epoch} | "
                "{learning_rate} | {learning_rate_decay} | {num_layers} | {hidden_dim} | "
                "{embedding_dim} | {dropout} | {batch_size} | {num_workers} |".format(**row)
            )
        lines.append("")
        lines.append("## Factor Signals")
        lines.append("")
        lines.append("Single-run candidates are not enough for causal conclusions, but these are the best observed values by factor.")
        for key in (
            "learning_rate",
            "learning_rate_decay",
            "num_layers",
            "hidden_dim",
            "embedding_dim",
            "dropout",
            "batch_size",
        ):
            summary = factor_summary(rows, key)
            if not summary:
                continue
            best = summary[0]
            lines.append(
                f"- `{key}`: best observed `{best[0]}` with min RMSE {best[2]:.6g} "
                f"(n={best[1]}, median {best[3]:.6g})."
            )
    if missing:
        lines.append("")
        lines.append("## Missing")
        lines.append("")
        for row in missing:
            lines.append(f"- `{row['run_name']}` / `{row['config_id']}`: {row['status']}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    args = parser.parse_args()

    manifest = args.sweep_dir / "manifest.tsv"
    rows = summarize_rows(read_manifest(manifest), args.results_root)
    summary_tsv = args.sweep_dir / "summary.tsv"
    summary_md = args.sweep_dir / "summary.md"
    write_tsv(rows, summary_tsv)
    write_markdown(rows, summary_md)
    print(summary_md.read_text())
    print(f"wrote {summary_tsv}")
    print(f"wrote {summary_md}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract best-checkpoint line and molecule embeddings for paper10 FT Morgan runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LONG_TABLE = REPO_ROOT / "results" / "lpm_paper10_results_long.tsv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "lpm_paper10_ft_morgan_learned_fixmol_best_embeddings"
DEFAULT_FAMILY = "finetune_morgan_learned_fixed_updated_embeddings"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_state(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        raw = torch.load(path, map_location="cpu", weights_only=False)

    metadata: dict[str, Any] = {"checkpoint_path": str(path)}
    if isinstance(raw, dict) and "state_dict" in raw:
        metadata["checkpoint_format"] = "lightning_checkpoint"
        metadata["epoch"] = raw.get("epoch")
        metadata["global_step"] = raw.get("global_step")
        return raw["state_dict"], metadata
    if isinstance(raw, tuple) and len(raw) == 3:
        model_id, model_args, state = raw
        metadata["checkpoint_format"] = "model_pt_tuple"
        metadata["model_id"] = model_id
        metadata["model_args"] = model_args
        return state, metadata
    raise ValueError(f"Unsupported checkpoint format: {path}")


def select_best_rows(long_table: Path, family: str) -> list[dict[str, str]]:
    by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in read_tsv(long_table):
        if row.get("model_family") != family:
            continue
        if row.get("status") != "ok" or not row.get("val_rmse") or not row.get("checkpoint_path"):
            continue
        checkpoint = Path(row["checkpoint_path"])
        if not checkpoint.is_file():
            continue
        by_dataset.setdefault(row["dataset_slug"], []).append(row)

    selected: list[dict[str, str]] = []
    for dataset_slug in sorted(by_dataset):
        rows = by_dataset[dataset_slug]
        selected.append(min(rows, key=lambda row: float(row["val_rmse"])))
    return selected


def config_path(config_id: str) -> Path:
    path = REPO_ROOT / "perturb_gym" / "configs" / "collection" / f"{config_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_data_sources(config_id: str) -> tuple[Path, list[str]]:
    config = yaml.safe_load(config_path(config_id).read_text())
    data_config = config["data_configs"][0]
    return Path(data_config["on_disk_shard_root"]), list(data_config["on_disk_data_sources"])


def collect_relevant_symbols(shard_root: Path, sources: list[str]) -> tuple[set[str], set[str], set[str]]:
    datasets: set[str] = set()
    contexts: set[str] = set()
    perturbations: set[str] = set()

    for source in sources:
        metadata_path = shard_root / source / "metadata.parquet"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = pl.read_parquet(metadata_path, columns=["datasets", "contexts", "perturbations"])
        for row in metadata.iter_rows(named=True):
            datasets.update(str(value) for value in row["datasets"])
            contexts.update(str(value) for value in row["contexts"])
            perturbations.update(str(value) for value in row["perturbations"])
    return datasets, contexts, perturbations


def subset_embeddings(
    state: dict[str, Any],
    symbol_key: str,
    weight_key: str,
    selected_symbols: set[str],
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    symbols = [str(symbol) for symbol in state[symbol_key]]
    symbol_to_code = {symbol: idx for idx, symbol in enumerate(symbols)}
    missing = sorted(selected_symbols - symbol_to_code.keys())
    kept = [symbol for symbol in symbols if symbol in selected_symbols]
    codes = [symbol_to_code[symbol] for symbol in kept]
    weights = state[weight_key].detach().cpu().numpy().astype(np.float32, copy=False)
    embeddings = weights[codes].copy()
    metadata = pd.DataFrame(
        {
            "code": np.arange(len(kept), dtype=np.int32),
            "checkpoint_code": np.asarray(codes, dtype=np.int32),
            "symbol": kept,
        }
    )
    return metadata, embeddings, missing


def write_embedding_bundle(
    output_dir: Path,
    stem: str,
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    pickle_column: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / f"{stem}_embeddings.npy"
    metadata_parquet = output_dir / f"{stem}_metadata.parquet"
    metadata_tsv = output_dir / f"{stem}_metadata.tsv"
    pickle_path = output_dir / f"df_{stem}.pkl"

    np.save(embeddings_path, embeddings)
    metadata.to_parquet(metadata_parquet, index=False)
    metadata.to_csv(metadata_tsv, sep="\t", index=False)

    frame = metadata.copy()
    frame[pickle_column] = embeddings.tolist()
    frame.to_pickle(pickle_path)

    return {
        "embeddings_npy": str(embeddings_path),
        "metadata_parquet": str(metadata_parquet),
        "metadata_tsv": str(metadata_tsv),
        "pickle": str(pickle_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-table", type=Path, default=DEFAULT_LONG_TABLE)
    parser.add_argument("--model-family", default=DEFAULT_FAMILY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_rows = select_best_rows(args.long_table, args.model_family)
    if not selected_rows:
        raise RuntimeError(f"No usable rows found for {args.model_family} in {args.long_table}")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    selection_rows: list[dict[str, Any]] = []

    for row in selected_rows:
        dataset_slug = row["dataset_slug"]
        dataset_name = row["dataset"]
        config_id = row["config_id"]
        checkpoint_path = Path(row["checkpoint_path"])
        dataset_dir = output_root / dataset_slug
        print(
            f"[extract] {dataset_slug}: seed={row['seed']} val={row['val_rmse']} "
            f"checkpoint={checkpoint_path.name}",
            flush=True,
        )

        shard_root, sources = load_data_sources(config_id)
        dataset_symbols, context_symbols, perturb_symbols = collect_relevant_symbols(shard_root, sources)
        state, checkpoint_metadata = load_state(checkpoint_path)

        line_metadata, line_embeddings, missing_contexts = subset_embeddings(
            state,
            symbol_key="context_symbols",
            weight_key="context_embedding_layer.weight",
            selected_symbols=context_symbols,
        )
        molecule_metadata, molecule_embeddings, missing_perturbations = subset_embeddings(
            state,
            symbol_key="perturb_symbols",
            weight_key="perturb_embedding_layer.weight",
            selected_symbols=perturb_symbols,
        )

        line_outputs = write_embedding_bundle(
            dataset_dir,
            stem="line",
            metadata=line_metadata,
            embeddings=line_embeddings,
            pickle_column="lpm_line_embeddings",
        )
        molecule_outputs = write_embedding_bundle(
            dataset_dir,
            stem="molecule",
            metadata=molecule_metadata,
            embeddings=molecule_embeddings,
            pickle_column="lpm_molecule_embeddings",
        )

        manifest = {
            "model_family": args.model_family,
            "dataset_slug": dataset_slug,
            "dataset": dataset_name,
            "seed": int(row["seed"]),
            "config_id": config_id,
            "val_rmse": float(row["val_rmse"]),
            "test_rmse": None if row.get("test_rmse", "") == "" else float(row["test_rmse"]),
            "checkpoint": checkpoint_metadata,
            "on_disk_shard_root": str(shard_root),
            "on_disk_data_sources": sources,
            "dataset_symbols": sorted(dataset_symbols),
            "n_relevant_lines": int(line_embeddings.shape[0]),
            "n_relevant_molecules": int(molecule_embeddings.shape[0]),
            "embedding_dim": int(line_embeddings.shape[1]) if line_embeddings.size else None,
            "missing_context_symbols": missing_contexts,
            "missing_perturbation_symbols": missing_perturbations,
            "outputs": {
                "line": line_outputs,
                "molecule": molecule_outputs,
            },
        }
        manifest_path = dataset_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        selection_rows.append(
            {
                "dataset_slug": dataset_slug,
                "dataset": dataset_name,
                "seed": row["seed"],
                "config_id": config_id,
                "val_rmse": row["val_rmse"],
                "test_rmse": row.get("test_rmse", ""),
                "checkpoint_path": str(checkpoint_path),
                "n_sources": len(sources),
                "n_lines": int(line_embeddings.shape[0]),
                "n_molecules": int(molecule_embeddings.shape[0]),
                "line_embeddings_npy": line_outputs["embeddings_npy"],
                "molecule_embeddings_npy": molecule_outputs["embeddings_npy"],
                "manifest_json": str(manifest_path),
                "missing_contexts": len(missing_contexts),
                "missing_perturbations": len(missing_perturbations),
            }
        )

    selection_path = output_root / "best_checkpoint_embedding_exports.tsv"
    write_tsv(
        selection_path,
        selection_rows,
        [
            "dataset_slug",
            "dataset",
            "seed",
            "config_id",
            "val_rmse",
            "test_rmse",
            "checkpoint_path",
            "n_sources",
            "n_lines",
            "n_molecules",
            "line_embeddings_npy",
            "molecule_embeddings_npy",
            "manifest_json",
            "missing_contexts",
            "missing_perturbations",
        ],
    )
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "model_family": args.model_family,
                "long_table": str(args.long_table),
                "datasets": len(selection_rows),
                "selection_tsv": str(selection_path),
            },
            indent=2,
        )
        + "\n"
    )

    print(f"[extract] wrote {selection_path}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create molecule-level val/test splits with cross-dataset training support.

For each dataset, molecules eligible for holdout must also occur in at least one
other selected dataset. A held-out molecule is never selected in every dataset
where it appears, so every held-out molecule keeps at least one training copy in
some other dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from build_multiout_plibdata import (
    CachePaths,
    load_config,
    make_source_specs,
    resolve_cache_paths,
)

DEFAULT_CONFIG = Path(
    "perturb_gym/configs/collection/lpm_multiout_all_data_plus_tahoe_novartis_h100_2x_bs4096_200epoch_lustre.yaml"
)
DEFAULT_OUTPUT_DIR = Path("results/cross_dataset_molecule_holdout_split")
DEFAULT_FALLBACK_CACHE_ROOT = Path(
    "/lustre/groups/ml01/workspace/olga.novitskaia/login_node_files/lpm_style/.plib_cache"
)
SAMPLE_COLUMNS = ["dataset", "context", "perturbation", "log_dose", "time"]
CONTROL_SYMBOL = "Control"
SPLIT_LOW_SHARED_BELOW = 20


def log(message: str) -> None:
    print(f"[create_cross_dataset_molecule_holdout_split] {message}", flush=True)


def percent(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def stable_shuffle(values: list[str], seed: int, salt: str) -> list[str]:
    salt_int = int(hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()[:16], 16)
    rng = random.Random(salt_int)
    values = list(values)
    rng.shuffle(values)
    return values


def fallback_cache_paths(paths: list[Path]) -> list[CachePaths]:
    out: list[CachePaths] = []
    for cache_root in paths:
        raw_root = cache_root / "raw_datasets"
        annotations_path = cache_root / "annotations" / "df_annot_split.parquet"
        plibdata_root = cache_root / "plibdata"
        if raw_root.is_dir() and annotations_path.is_file():
            out.append(
                CachePaths(
                    cache_root=cache_root,
                    raw_root=raw_root,
                    annotations_path=annotations_path,
                    plibdata_root=plibdata_root,
                )
            )
    return out


def read_source_samples(files: tuple[Path, ...], context: str, source: str) -> pl.DataFrame:
    return (
        pl.scan_parquet([str(path) for path in files])
        .filter(pl.col("context") == context)
        .select(SAMPLE_COLUMNS)
        .unique(maintain_order=False)
        .with_columns(source=pl.lit(source))
        .collect()
    )


def molecule_tokens(perturbation: str) -> list[str]:
    return [token for token in str(perturbation).split("+") if token and token != CONTROL_SYMBOL]


def collect_samples(config_path: Path, args: argparse.Namespace) -> tuple[pl.DataFrame, list[str]]:
    plibdata_root, sources, _config = load_config(config_path)
    if args.cache_root is None and args.raw_root is None and plibdata_root.parent.name == "plibdata_multiout":
        args.cache_root = plibdata_root.parent.parent
    cache_root, raw_root, annotations_path = resolve_cache_paths(args, plibdata_root)
    reference_plibdata_root = args.reference_plibdata_root or cache_root / "plibdata"
    cache_paths = [
        CachePaths(
            cache_root=cache_root,
            raw_root=raw_root,
            annotations_path=annotations_path,
            plibdata_root=reference_plibdata_root,
        )
    ]
    fallback_roots = list(args.fallback_cache_root or [])
    if DEFAULT_FALLBACK_CACHE_ROOT.is_dir():
        fallback_roots.append(DEFAULT_FALLBACK_CACHE_ROOT)
    cache_paths.extend(fallback_cache_paths(fallback_roots))
    specs = make_source_specs(cache_paths, sources)

    frames: list[pl.DataFrame] = []
    log(f"config={config_path}")
    log(f"sources={len(specs)}")
    for index, spec in enumerate(specs, start=1):
        log(f"{index}/{len(specs)} {spec.source}")
        source_samples = read_source_samples(spec.files, spec.context, spec.source)
        if not source_samples.is_empty():
            frames.append(source_samples)
    if not frames:
        raise RuntimeError("No samples found for selected sources.")
    return pl.concat(frames, how="vertical").unique(maintain_order=False), sources


def build_molecule_indexes(samples: pl.DataFrame) -> tuple[
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, dict[str, int]],
    dict[str, int],
    dict[str, int],
]:
    samples = samples.with_row_index("_sample_id")
    sample_molecules = (
        samples.select(["dataset", "_sample_id", "perturbation"])
        .with_columns(pl.col("perturbation").cast(pl.Utf8).str.split("+").alias("molecule"))
        .explode("molecule")
        .filter((pl.col("molecule") != CONTROL_SYMBOL) & pl.col("molecule").is_not_null())
        .select(["dataset", "_sample_id", "molecule"])
        .unique(maintain_order=False)
    )

    molecules_by_dataset: dict[str, set[str]] = defaultdict(set)
    datasets_by_molecule: dict[str, set[str]] = defaultdict(set)
    sample_counts_by_dataset_molecule: dict[str, dict[str, int]] = defaultdict(dict)

    for row in sample_molecules.select(["dataset", "molecule"]).unique().iter_rows(named=True):
        dataset = str(row["dataset"])
        molecule = str(row["molecule"])
        molecules_by_dataset[dataset].add(molecule)
        datasets_by_molecule[molecule].add(dataset)

    counts = sample_molecules.group_by(["dataset", "molecule"]).agg(pl.col("_sample_id").n_unique().alias("n"))
    for row in counts.iter_rows(named=True):
        sample_counts_by_dataset_molecule[str(row["dataset"])][str(row["molecule"])] = int(row["n"])

    total_samples = {
        str(row["dataset"]): int(row["len"])
        for row in samples.group_by("dataset").len().iter_rows(named=True)
    }
    perturbed_samples = {
        str(row["dataset"]): int(row["n"])
        for row in sample_molecules.group_by("dataset")
        .agg(pl.col("_sample_id").n_unique().alias("n"))
        .iter_rows(named=True)
    }
    return (
        molecules_by_dataset,
        datasets_by_molecule,
        sample_counts_by_dataset_molecule,
        total_samples,
        perturbed_samples,
    )


def can_hold_out(
    dataset: str,
    molecule: str,
    selected_datasets_by_molecule: dict[str, set[str]],
    datasets_by_molecule: dict[str, set[str]],
) -> bool:
    selected = selected_datasets_by_molecule[molecule] | {dataset}
    return len(datasets_by_molecule[molecule] - selected) > 0


def partition_low_shared_by_samples(
    dataset: str,
    molecules: list[str],
    sample_counts: dict[str, int],
) -> tuple[set[str], set[str]]:
    val: set[str] = set()
    test: set[str] = set()
    val_samples = 0
    test_samples = 0
    ordered = sorted(molecules, key=lambda molecule: (-sample_counts.get(molecule, 0), molecule))
    for molecule in ordered:
        if val_samples <= test_samples:
            val.add(molecule)
            val_samples += sample_counts.get(molecule, 0)
        else:
            test.add(molecule)
            test_samples += sample_counts.get(molecule, 0)
    return val, test


def choose_holdouts(
    molecules_by_dataset: dict[str, set[str]],
    datasets_by_molecule: dict[str, set[str]],
    sample_counts_by_dataset_molecule: dict[str, dict[str, int]],
    seed: int,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    shared_by_dataset = {
        dataset: {molecule for molecule in molecules if len(datasets_by_molecule[molecule] - {dataset}) > 0}
        for dataset, molecules in molecules_by_dataset.items()
    }
    selected_datasets_by_molecule: dict[str, set[str]] = defaultdict(set)
    val_by_dataset: dict[str, set[str]] = defaultdict(set)
    test_by_dataset: dict[str, set[str]] = defaultdict(set)
    strategy_by_dataset: dict[str, str] = {}

    dataset_order = sorted(shared_by_dataset, key=lambda dataset: (len(shared_by_dataset[dataset]), dataset))
    for dataset in dataset_order:
        shared = shared_by_dataset[dataset]
        counts = sample_counts_by_dataset_molecule.get(dataset, {})
        if len(shared) < SPLIT_LOW_SHARED_BELOW:
            strategy_by_dataset[dataset] = "low_shared_sample_balanced"
            eligible = [
                molecule
                for molecule in sorted(shared)
                if can_hold_out(dataset, molecule, selected_datasets_by_molecule, datasets_by_molecule)
            ]
            val, test = partition_low_shared_by_samples(dataset, eligible, counts)
        else:
            strategy_by_dataset[dataset] = "ten_percent_shared_molecules"
            target_each = math.floor(0.10 * len(shared))
            val = set()
            test = set()
            candidates = stable_shuffle(sorted(shared), seed, dataset)
            for molecule in candidates:
                if len(val) >= target_each:
                    break
                if can_hold_out(dataset, molecule, selected_datasets_by_molecule, datasets_by_molecule):
                    val.add(molecule)
                    selected_datasets_by_molecule[molecule].add(dataset)
            for molecule in candidates:
                if len(test) >= target_each:
                    break
                if molecule in val:
                    continue
                if can_hold_out(dataset, molecule, selected_datasets_by_molecule, datasets_by_molecule):
                    test.add(molecule)
                    selected_datasets_by_molecule[molecule].add(dataset)
            val_by_dataset[dataset] = val
            test_by_dataset[dataset] = test
            continue

        for molecule in val | test:
            selected_datasets_by_molecule[molecule].add(dataset)
        val_by_dataset[dataset] = val
        test_by_dataset[dataset] = test

    return val_by_dataset, test_by_dataset, strategy_by_dataset


def assign_splits(samples: pl.DataFrame, val_by_dataset: dict[str, set[str]], test_by_dataset: dict[str, set[str]]) -> pl.DataFrame:
    val_keys = {
        f"{dataset}\t{molecule}"
        for dataset, molecules in val_by_dataset.items()
        for molecule in molecules
    }
    test_keys = {
        f"{dataset}\t{molecule}"
        for dataset, molecules in test_by_dataset.items()
        for molecule in molecules
    }

    def split_for_row(row: dict[str, Any]) -> str:
        dataset = str(row["dataset"])
        keys = {f"{dataset}\t{molecule}" for molecule in molecule_tokens(str(row["perturbation"]))}
        if keys & test_keys:
            return "test"
        if keys & val_keys:
            return "val"
        return "train"

    return samples.with_columns(
        split=pl.struct(["dataset", "perturbation"]).map_elements(split_for_row, return_dtype=pl.Utf8)
    ).select(SAMPLE_COLUMNS + ["split"])


def summarize(
    annotation: pl.DataFrame,
    molecules_by_dataset: dict[str, set[str]],
    datasets_by_molecule: dict[str, set[str]],
    val_by_dataset: dict[str, set[str]],
    test_by_dataset: dict[str, set[str]],
    strategy_by_dataset: dict[str, str],
    total_samples: dict[str, int],
    perturbed_samples: dict[str, int],
) -> pl.DataFrame:
    split_counts = {
        (str(row["dataset"]), str(row["split"])): int(row["len"])
        for row in annotation.group_by(["dataset", "split"]).len().iter_rows(named=True)
    }
    rows = []
    for dataset in sorted(molecules_by_dataset):
        shared = {molecule for molecule in molecules_by_dataset[dataset] if len(datasets_by_molecule[molecule] - {dataset}) > 0}
        val = val_by_dataset.get(dataset, set())
        test = test_by_dataset.get(dataset, set())
        selected = val | test
        violations = [
            molecule
            for molecule in selected
            if not (datasets_by_molecule[molecule] - {other_dataset for other_dataset in datasets_by_molecule[molecule] if molecule in val_by_dataset.get(other_dataset, set()) or molecule in test_by_dataset.get(other_dataset, set())})
        ]
        rows.append(
            {
                "dataset": dataset,
                "strategy": strategy_by_dataset.get(dataset, ""),
                "total_samples": total_samples.get(dataset, 0),
                "perturbed_samples": perturbed_samples.get(dataset, 0),
                "total_molecules": len(molecules_by_dataset[dataset]),
                "shared_molecules": len(shared),
                "val_molecules": len(val),
                "test_molecules": len(test),
                "pct_shared_molecules_val": percent(len(val), len(shared)),
                "pct_shared_molecules_test": percent(len(test), len(shared)),
                "train_samples": split_counts.get((dataset, "train"), 0),
                "val_samples": split_counts.get((dataset, "val"), 0),
                "test_samples": split_counts.get((dataset, "test"), 0),
                "pct_samples_val": percent(split_counts.get((dataset, "val"), 0), total_samples.get(dataset, 0)),
                "pct_samples_test": percent(split_counts.get((dataset, "test"), 0), total_samples.get(dataset, 0)),
                "training_copy_violations": len(violations),
            }
        )
    return pl.DataFrame(rows).sort("dataset")


def write_heldout_table(
    path: Path,
    val_by_dataset: dict[str, set[str]],
    test_by_dataset: dict[str, set[str]],
    datasets_by_molecule: dict[str, set[str]],
    sample_counts_by_dataset_molecule: dict[str, dict[str, int]],
) -> None:
    rows = []
    selected_by_molecule: dict[str, set[str]] = defaultdict(set)
    for dataset, molecules in val_by_dataset.items():
        for molecule in molecules:
            selected_by_molecule[molecule].add(dataset)
    for dataset, molecules in test_by_dataset.items():
        for molecule in molecules:
            selected_by_molecule[molecule].add(dataset)

    for split_name, by_dataset in (("val", val_by_dataset), ("test", test_by_dataset)):
        for dataset, molecules in by_dataset.items():
            for molecule in sorted(molecules):
                training_datasets = sorted(datasets_by_molecule[molecule] - selected_by_molecule[molecule])
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split_name,
                        "molecule": molecule,
                        "samples_in_dataset": sample_counts_by_dataset_molecule.get(dataset, {}).get(molecule, 0),
                        "other_training_datasets": ",".join(training_datasets),
                    }
                )
    pl.DataFrame(rows).sort(["dataset", "split", "molecule"]).write_csv(path, separator="\t")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--reference-plibdata-root", type=Path, default=None)
    parser.add_argument("--fallback-cache-root", action="append", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples, sources = collect_samples(args.config, args)
    (
        molecules_by_dataset,
        datasets_by_molecule,
        sample_counts_by_dataset_molecule,
        total_samples,
        perturbed_samples,
    ) = build_molecule_indexes(samples)
    val_by_dataset, test_by_dataset, strategy_by_dataset = choose_holdouts(
        molecules_by_dataset=molecules_by_dataset,
        datasets_by_molecule=datasets_by_molecule,
        sample_counts_by_dataset_molecule=sample_counts_by_dataset_molecule,
        seed=args.seed,
    )
    annotation = assign_splits(samples, val_by_dataset, test_by_dataset)
    summary = summarize(
        annotation=annotation,
        molecules_by_dataset=molecules_by_dataset,
        datasets_by_molecule=datasets_by_molecule,
        val_by_dataset=val_by_dataset,
        test_by_dataset=test_by_dataset,
        strategy_by_dataset=strategy_by_dataset,
        total_samples=total_samples,
        perturbed_samples=perturbed_samples,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = args.output_dir / "df_annot_split.parquet"
    annotation.write_parquet(annotation_path, compression="zstd", statistics=True)
    annotation.write_csv(args.output_dir / "df_annot_split.tsv", separator="\t")
    summary.write_csv(args.output_dir / "summary.tsv", separator="\t")
    write_heldout_table(
        args.output_dir / "heldout_molecules.tsv",
        val_by_dataset,
        test_by_dataset,
        datasets_by_molecule,
        sample_counts_by_dataset_molecule,
    )
    manifest = {
        "config": str(args.config),
        "sources": sources,
        "seed": args.seed,
        "split_low_shared_below": SPLIT_LOW_SHARED_BELOW,
        "outputs": {
            "annotation_parquet": str(annotation_path),
            "annotation_tsv": str(args.output_dir / "df_annot_split.tsv"),
            "summary_tsv": str(args.output_dir / "summary.tsv"),
            "heldout_molecules_tsv": str(args.output_dir / "heldout_molecules.tsv"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "source_config_snapshot.yaml").write_text(yaml.safe_dump({"sources": sources}, sort_keys=False))
    log(f"wrote {annotation_path}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

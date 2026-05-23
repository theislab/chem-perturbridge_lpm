#!/usr/bin/env python3
"""Measure which molecules can be held out per dataset while seen elsewhere.

For each raw dataset represented by a multi-output training config, this script
counts molecules that occur in that dataset and in at least one other selected
dataset. Those are candidates for dataset-specific molecule holdout splits that
still leave the molecule observed in another dataset's training set.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl

from build_multiout_plibdata import (
    CachePaths,
    load_config,
    make_source_specs,
    resolve_cache_paths,
)

DEFAULT_CONFIG = Path(
    "perturb_gym/configs/collection/"
    "lpm_multiout_all_tahoe_novartis_sciplex_no_dili_vcpi0001_cigs_mce_h100_2x_bs4096_200epoch_lustre.yaml"
)
DEFAULT_FALLBACK_CACHE_ROOT = Path(
    "/lustre/groups/ml01/workspace/olga.novitskaia/login_node_files/lpm_style/.plib_cache"
)
DEFAULT_OUTPUT = Path("results/cross_dataset_molecule_overlap.tsv")
SAMPLE_COLUMNS = ["dataset", "context", "perturbation", "log_dose", "time"]
CONTROL_SYMBOL = "Control"


def log(message: str) -> None:
    print(f"[explore_cross_dataset_molecule_overlap] {message}", flush=True)


def percent(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


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


def read_source_samples(files: tuple[Path, ...], context: str) -> pl.DataFrame:
    return (
        pl.scan_parquet([str(path) for path in files])
        .filter(pl.col("context") == context)
        .select(SAMPLE_COLUMNS)
        .unique(maintain_order=False)
        .collect()
    )


def dataset_sample_stats(samples: pl.DataFrame) -> tuple[dict[str, int], dict[str, int], dict[str, set[str]], dict[str, set[Any]]]:
    total_samples: dict[str, int] = {}
    perturbed_samples: dict[str, int] = {}
    molecules_by_dataset: dict[str, set[str]] = defaultdict(set)
    sample_ids_by_dataset_molecule: dict[str, set[Any]] = defaultdict(set)

    if samples.is_empty():
        return total_samples, perturbed_samples, molecules_by_dataset, sample_ids_by_dataset_molecule

    samples = samples.with_row_index("_sample_id")
    totals = samples.group_by("dataset").len().iter_rows(named=True)
    total_samples = {str(row["dataset"]): int(row["len"]) for row in totals}

    sample_molecules = (
        samples.select(["dataset", "_sample_id", "perturbation"])
        .with_columns(pl.col("perturbation").cast(pl.Utf8).str.split("+").alias("molecule"))
        .explode("molecule")
        .filter((pl.col("molecule") != CONTROL_SYMBOL) & pl.col("molecule").is_not_null())
        .select(["dataset", "_sample_id", "molecule"])
        .unique(maintain_order=False)
    )
    if sample_molecules.is_empty():
        return total_samples, perturbed_samples, molecules_by_dataset, sample_ids_by_dataset_molecule

    perturbed_counts = sample_molecules.group_by("dataset").agg(pl.col("_sample_id").n_unique().alias("n"))
    perturbed_samples = {str(row["dataset"]): int(row["n"]) for row in perturbed_counts.iter_rows(named=True)}

    for row in sample_molecules.select(["dataset", "molecule"]).unique().iter_rows(named=True):
        molecules_by_dataset[str(row["dataset"])].add(str(row["molecule"]))

    for row in sample_molecules.iter_rows(named=True):
        key = f"{row['dataset']}\t{row['molecule']}"
        sample_ids_by_dataset_molecule[key].add(row["_sample_id"])

    return total_samples, perturbed_samples, molecules_by_dataset, sample_ids_by_dataset_molecule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--reference-plibdata-root", type=Path, default=None)
    parser.add_argument("--fallback-cache-root", action="append", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plibdata_root, sources, _config = load_config(args.config)
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

    log(f"config={args.config}")
    log(f"sources={len(specs)}")
    samples_by_dataset: dict[str, list[pl.DataFrame]] = defaultdict(list)
    source_count_by_dataset: dict[str, set[str]] = defaultdict(set)
    contexts_by_dataset: dict[str, set[str]] = defaultdict(set)

    for index, spec in enumerate(specs, start=1):
        log(f"{index}/{len(specs)} {spec.source}")
        samples = read_source_samples(spec.files, spec.context)
        if samples.is_empty():
            continue
        for dataset in samples["dataset"].unique().to_list():
            dataset_name = str(dataset)
            source_count_by_dataset[dataset_name].add(spec.source)
            contexts_by_dataset[dataset_name].add(spec.context)
            samples_by_dataset[dataset_name].append(samples.filter(pl.col("dataset") == dataset))

    all_total_samples: dict[str, int] = {}
    all_perturbed_samples: dict[str, int] = {}
    molecules_by_dataset: dict[str, set[str]] = defaultdict(set)
    sample_ids_by_dataset_molecule: dict[str, set[Any]] = defaultdict(set)
    for dataset, frames in samples_by_dataset.items():
        dataset_samples = pl.concat(frames, how="vertical").unique(maintain_order=False)
        total_samples, perturbed_samples, dataset_molecules, sample_ids = dataset_sample_stats(dataset_samples)
        all_total_samples[dataset] = total_samples.get(dataset, 0)
        all_perturbed_samples[dataset] = perturbed_samples.get(dataset, 0)
        molecules_by_dataset[dataset] = dataset_molecules.get(dataset, set())
        sample_ids_by_dataset_molecule.update(sample_ids)

    datasets_by_molecule: dict[str, set[str]] = defaultdict(set)
    for dataset, molecules in molecules_by_dataset.items():
        for molecule in molecules:
            datasets_by_molecule[molecule].add(dataset)

    rows = []
    for dataset in sorted(molecules_by_dataset):
        total_molecules = len(molecules_by_dataset[dataset])
        cross_dataset_molecules = {
            molecule
            for molecule in molecules_by_dataset[dataset]
            if len(datasets_by_molecule[molecule] - {dataset}) > 0
        }
        cross_sample_ids: set[Any] = set()
        for molecule in cross_dataset_molecules:
            cross_sample_ids.update(sample_ids_by_dataset_molecule.get(f"{dataset}\t{molecule}", set()))
        n_cross_samples = len(cross_sample_ids)
        total_samples = all_total_samples.get(dataset, 0)
        perturbed_samples = all_perturbed_samples.get(dataset, 0)
        rows.append(
            {
                "dataset": dataset,
                "selected_sources": len(source_count_by_dataset.get(dataset, set())),
                "contexts": len(contexts_by_dataset.get(dataset, set())),
                "total_samples": total_samples,
                "perturbed_samples": perturbed_samples,
                "total_molecules": total_molecules,
                "molecules_present_in_other_datasets": len(cross_dataset_molecules),
                "pct_molecules_present_in_other_datasets": percent(len(cross_dataset_molecules), total_molecules),
                "samples_with_molecules_present_in_other_datasets": n_cross_samples,
                "pct_all_samples_with_molecules_present_in_other_datasets": percent(n_cross_samples, total_samples),
                "pct_perturbed_samples_with_molecules_present_in_other_datasets": percent(
                    n_cross_samples, perturbed_samples
                ),
            }
        )

    result = pl.DataFrame(rows).sort("dataset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(args.output, separator="\t")
    log(f"wrote {args.output}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

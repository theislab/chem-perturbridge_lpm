#!/usr/bin/env python3
"""Build sample-level multi-output shards from the same inputs as notebook 03.

README.md chapter 2 points at the three LPM-style notebooks. The final shard
writer, ``notebooks/03_LPM_style_write_shards.ipynb``, starts from:

  * ``.plib_cache/raw_datasets/<dataset_folder>/*.parquet``
  * ``.plib_cache/annotations/df_annot_split.parquet``

and writes long-row scalar shards under ``.plib_cache/plibdata``. This script
starts from those same files, but writes one row per sample with ragged readout
targets for a multi-output model:

  dataset_code, context_code, perturbation_codes, log_dose, time, split,
  readout_codes, values, n_values

The YAML config is used only to choose the exact source list from the recent
training run and to locate the cache root. Existing split labels are preserved;
no resplitting or subsampling is performed.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import yaml

DEFAULT_CONFIG = Path("perturb_gym/configs/collection/lpm_modified_all_data_1epoch_lustre.yaml")
DEFAULT_OUTPUT_ROOT = Path(".plib_cache/plibdata_multiout/lpm_modified_all_data_1epoch_lustre")
JOIN_KEYS = ["dataset", "context", "perturbation"]
SAMPLE_COLUMNS = ["dataset", "context", "perturbation", "log_dose", "time"]
RAW_COLUMNS = ["dataset", "context", "perturbation", "log_dose", "time", "readout", "value"]
SPLIT_ORDER = ("train", "val", "test")
CONTROL_SYMBOL = "Control"


@dataclass(frozen=True)
class CachePaths:
    cache_root: Path
    raw_root: Path
    annotations_path: Path
    plibdata_root: Path


@dataclass(frozen=True)
class SourceSpec:
    source: str
    dataset_folder: str
    context: str
    files: tuple[Path, ...]
    cache_root: Path
    annotations_path: Path
    plibdata_root: Path


@dataclass
class Vocab:
    dataset_symbols: list[str]
    context_symbols: list[str]
    perturb_symbols: list[str]
    readout_symbols: list[str]


@dataclass
class SplitStats:
    raw_scalar_values: int = 0
    joined_scalar_values: int = 0
    dropped_without_split: int = 0
    output_samples: int = 0
    output_shards: int = 0


@dataclass
class SourceStats:
    source: str
    raw_scalar_values: int = 0
    joined_scalar_values: int = 0
    dropped_without_split: int = 0
    scalar_values_written: int = 0
    output_samples: int = 0
    output_shards: int = 0
    split_stats: dict[str, SplitStats] = field(default_factory=dict)


def log(message: str) -> None:
    print(f"[build_multiout_plibdata] {message}", flush=True)


def load_config(config_path: Path) -> tuple[Path, list[str], dict[str, Any]]:
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle)
    data_configs = config.get("data_configs", [])
    if len(data_configs) != 1:
        raise ValueError(f"Expected exactly one data config in {config_path}, found {len(data_configs)}")
    data_config = data_configs[0]
    plibdata_root = Path(data_config["on_disk_shard_root"])
    sources = list(data_config["on_disk_data_sources"])
    return plibdata_root, sources, config


def resolve_cache_paths(args: argparse.Namespace, plibdata_root: Path) -> tuple[Path, Path, Path]:
    cache_root = args.cache_root or plibdata_root.parent
    raw_root = args.raw_root or cache_root / "raw_datasets"
    annotations_path = args.annotations or cache_root / "annotations" / "df_annot_split.parquet"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Missing raw dataset root used by notebook 03: {raw_root}")
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Missing split annotation file used by notebook 03: {annotations_path}")
    return cache_root, raw_root, annotations_path


def raw_dataset_folders(raw_root: Path) -> list[str]:
    return sorted(path.name for path in raw_root.iterdir() if path.is_dir())


def split_source_name(source: str, folders: list[str]) -> tuple[str, str]:
    matches = [folder for folder in folders if source.startswith(f"{folder}_")]
    if not matches:
        raise ValueError(f"Could not map source '{source}' to a raw_datasets folder.")
    dataset_folder = max(matches, key=len)
    context = source[len(dataset_folder) + 1 :]
    return dataset_folder, context


def source_files(raw_root: Path, dataset_folder: str, context: str) -> tuple[Path, ...]:
    dataset_dir = raw_root / dataset_folder
    files = sorted(dataset_dir.glob("*.parquet"))
    candidates = [path for path in files if path.stem == context or path.stem.startswith(f"{context}_")]
    return tuple(candidates or files)


def source_context(files: tuple[Path, ...], context: str) -> str:
    if not files:
        return context
    contexts = (
        pl.scan_parquet([str(path) for path in files])
        .select(pl.col("context").unique())
        .collect()["context"]
        .drop_nulls()
        .to_list()
    )
    if context in contexts:
        return context
    normalized = context.replace("_", " ")
    if normalized in contexts:
        return normalized
    if len(contexts) == 1:
        return str(contexts[0])
    return context


def make_source_specs(cache_paths: list[CachePaths], sources: list[str]) -> list[SourceSpec]:
    folder_lists = [(paths, raw_dataset_folders(paths.raw_root)) for paths in cache_paths]
    specs: list[SourceSpec] = []
    for source in sources:
        errors: list[str] = []
        for paths, folders in folder_lists:
            try:
                dataset_folder, context = split_source_name(source, folders)
                files = source_files(paths.raw_root, dataset_folder, context)
            except ValueError as exc:
                errors.append(f"{paths.raw_root}: {exc}")
                continue
            if not files:
                errors.append(f"{paths.raw_root / dataset_folder}: no raw parquet files")
                continue
            specs.append(
                SourceSpec(
                    source=source,
                    dataset_folder=dataset_folder,
                    context=source_context(files, context),
                    files=files,
                    cache_root=paths.cache_root,
                    annotations_path=paths.annotations_path,
                    plibdata_root=paths.plibdata_root,
                )
            )
            break
        else:
            raise FileNotFoundError(
                f"Could not resolve source {source} in any configured cache root: " + "; ".join(errors)
            )
    return specs


def _vocab_frame(symbols: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"symbol": symbols, "code": list(range(len(symbols)))})


def _read_vocab_file(path: Path) -> list[str]:
    return pl.read_parquet(path).sort("code")["symbol"].to_list()


def load_vocab(output_root: Path) -> Vocab | None:
    vocab_dir = output_root / "vocab"
    paths = {
        "dataset_symbols": vocab_dir / "dataset_vocab.parquet",
        "context_symbols": vocab_dir / "context_vocab.parquet",
        "perturb_symbols": vocab_dir / "perturb_vocab.parquet",
        "readout_symbols": vocab_dir / "readout_vocab.parquet",
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    return Vocab(**{key: _read_vocab_file(path) for key, path in paths.items()})


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def atomic_write_parquet(df: pl.DataFrame, path: Path, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    df.write_parquet(tmp, compression=compression, statistics=True)
    tmp.replace(path)


def source_list_text(sources: list[str]) -> str:
    return "\n".join(["index\tsource"] + [f"{idx}\t{source}" for idx, source in enumerate(sources)]) + "\n"


def validate_existing_sources(output_root: Path, sources: list[str]) -> None:
    sources_path = output_root / "sources.tsv"
    if not sources_path.exists():
        return
    expected = source_list_text(sources)
    existing = sources_path.read_text()
    if existing == expected:
        return
    raise RuntimeError(
        f"{output_root} already has a sources.tsv from a different source list. "
        "Use a distinct on_disk_shard_root for each dataset subset; refusing to overwrite."
    )


def write_vocab(output_root: Path, vocab: Vocab, compression: str) -> None:
    vocab_dir = output_root / "vocab"
    atomic_write_parquet(_vocab_frame(vocab.dataset_symbols), vocab_dir / "dataset_vocab.parquet", compression)
    atomic_write_parquet(_vocab_frame(vocab.context_symbols), vocab_dir / "context_vocab.parquet", compression)
    atomic_write_parquet(_vocab_frame(vocab.perturb_symbols), vocab_dir / "perturb_vocab.parquet", compression)
    atomic_write_parquet(_vocab_frame(vocab.readout_symbols), vocab_dir / "readout_vocab.parquet", compression)


def build_vocab_from_raw(specs: list[SourceSpec]) -> Vocab:
    datasets: set[str] = set()
    contexts: set[str] = set()
    perturbations: set[str] = set()
    readouts: set[str] = set()

    for spec in specs:
        log(f"vocab scan: {spec.source} ({len(spec.files)} raw file(s))")
        for path in spec.files:
            df = (
                pl.scan_parquet(path)
                .filter(pl.col("context") == spec.context)
                .select(["dataset", "context", "perturbation", "readout"])
                .collect()
            )
            if df.is_empty():
                continue
            datasets.update(str(x) for x in df["dataset"].unique().to_list())
            contexts.update(str(x) for x in df["context"].unique().to_list())
            readouts.update(str(x) for x in df["readout"].unique().to_list())
            for perturbation in df["perturbation"].unique().to_list():
                perturbations.update(str(perturbation).split("+"))

    perturb_symbols = sorted(perturbations)
    if CONTROL_SYMBOL in perturb_symbols:
        perturb_symbols.remove(CONTROL_SYMBOL)
    perturb_symbols = [CONTROL_SYMBOL] + perturb_symbols

    return Vocab(
        dataset_symbols=sorted(datasets),
        context_symbols=sorted(contexts),
        perturb_symbols=perturb_symbols,
        readout_symbols=sorted(readouts),
    )


def get_or_build_vocab(output_root: Path, specs: list[SourceSpec], compression: str) -> Vocab:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".vocab.lock"
    with open(lock_path, "w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        vocab = load_vocab(output_root)
        if vocab is not None:
            log("loaded existing vocab")
            return vocab
        log("building vocab from notebook raw_datasets inputs")
        vocab = build_vocab_from_raw(specs)
        write_vocab(output_root, vocab, compression)
        log(
            "wrote vocab: "
            f"datasets={len(vocab.dataset_symbols)}, contexts={len(vocab.context_symbols)}, "
            f"perturbations={len(vocab.perturb_symbols)}, readouts={len(vocab.readout_symbols)}"
        )
        return vocab


def write_dataset_scaffolding(
    output_root: Path,
    config_path: Path,
    config: dict[str, Any],
    cache_paths: list[CachePaths],
    source_specs: list[SourceSpec],
    sources: list[str],
    target_values_per_shard: int,
    value_dtype: str,
    float_feature_dtype: str,
    include_symbols: bool,
) -> None:
    validate_existing_sources(output_root, sources)
    atomic_write_text(output_root / "sources.tsv", source_list_text(sources))
    primary_paths = cache_paths[0]
    source_input_cache_roots = {spec.source: str(spec.cache_root) for spec in source_specs}
    source_reference_plibdata_roots = {spec.source: str(spec.plibdata_root) for spec in source_specs}

    manifest = {
        "format": "lpm_multiout_plibdata",
        "format_version": 1,
        "created_from_config": str(config_path),
        "input_cache_root": str(primary_paths.cache_root),
        "input_raw_datasets_root": str(primary_paths.raw_root),
        "input_annotations_split": str(primary_paths.annotations_path),
        "reference_plibdata_root_for_count_validation": str(primary_paths.plibdata_root),
        "input_cache_roots": sorted({str(paths.cache_root) for paths in cache_paths}),
        "source_input_cache_roots": source_input_cache_roots,
        "source_reference_plibdata_roots": source_reference_plibdata_roots,
        "source_count": len(sources),
        "join_keys_matching_notebook_03": JOIN_KEYS,
        "sample_columns": SAMPLE_COLUMNS,
        "output_columns": [
            "dataset_code",
            "context_code",
            "perturbation_codes",
            "log_dose",
            "time",
            "split",
            "readout_codes",
            "values",
            "n_values",
        ],
        "target_values_per_shard": target_values_per_shard,
        "value_dtype": value_dtype,
        "float_feature_dtype": float_feature_dtype,
        "include_symbols": include_symbols,
        "same_data_contract": (
            "Starts from raw_datasets/*.parquet and annotations/df_annot_split.parquet, "
            "the same files consumed by notebooks/03_LPM_style_write_shards.ipynb. "
            "The YAML config only restricts processing to the exact source folders used "
            "by the recent training run."
        ),
    }
    atomic_write_text(output_root / "dataset_manifest.yaml", yaml.safe_dump(manifest, sort_keys=False))
    atomic_write_text(output_root / "source_config_snapshot.yaml", yaml.safe_dump(config, sort_keys=False))


def _encode_df(df: pl.DataFrame, vocab: Vocab, value_dtype: pl.DataType, feature_dtype: pl.DataType) -> pl.DataFrame:
    perturb_symbols = pl.Series("symbol", vocab.perturb_symbols)
    perturb_codes = pl.Series("code", list(range(len(vocab.perturb_symbols))), dtype=pl.Int64)
    return df.with_columns(
        dataset_code=pl.col("dataset")
        .replace_strict(vocab.dataset_symbols, list(range(len(vocab.dataset_symbols))))
        .cast(pl.Int64),
        context_code=pl.col("context")
        .replace_strict(vocab.context_symbols, list(range(len(vocab.context_symbols))))
        .cast(pl.Int64),
        perturbation_codes=pl.col("perturbation")
        .str.split("+")
        .list.eval(pl.element().replace_strict(perturb_symbols, perturb_codes).cast(pl.Int64)),
        readout_code=pl.col("readout")
        .replace_strict(vocab.readout_symbols, list(range(len(vocab.readout_symbols))))
        .cast(pl.Int64),
        log_dose=pl.col("log_dose").cast(feature_dtype),
        time=pl.col("time").cast(feature_dtype),
        value=pl.col("value").cast(value_dtype),
    )


def group_to_multiout(
    df: pl.DataFrame,
    split: str,
    vocab: Vocab,
    value_dtype: pl.DataType,
    feature_dtype: pl.DataType,
    include_symbols: bool,
) -> pl.DataFrame:
    if df.is_empty():
        return df

    encoded = _encode_df(df, vocab, value_dtype, feature_dtype)
    grouped = encoded.group_by(
        ["dataset_code", "context_code", "perturbation", "log_dose", "time"],
        maintain_order=True,
    ).agg(
        perturbation_codes=pl.col("perturbation_codes").first(),
        readout_codes=pl.col("readout_code"),
        values=pl.col("value"),
        split=pl.lit(split),
        **(
            {
                "dataset": pl.col("dataset").first(),
                "context": pl.col("context").first(),
            }
            if include_symbols
            else {}
        ),
    )
    grouped = grouped.with_columns(n_values=pl.col("values").list.len().cast(pl.UInt32))

    columns = [
        "dataset_code",
        "context_code",
        "perturbation_codes",
        "log_dose",
        "time",
        "split",
        "readout_codes",
        "values",
        "n_values",
    ]
    if include_symbols:
        columns = ["dataset", "context", "perturbation"] + columns
    return grouped.select(columns)


def metadata_row(source: str, rel_path: str, shard_df: pl.DataFrame, vocab: Vocab) -> dict[str, Any]:
    dataset_codes = sorted(set(shard_df["dataset_code"].to_list()))
    context_codes = sorted(set(shard_df["context_code"].to_list()))
    perturb_codes = sorted(set(shard_df.select(pl.col("perturbation_codes").explode())["perturbation_codes"].to_list()))
    readout_codes = sorted(set(shard_df.select(pl.col("readout_codes").explode())["readout_codes"].to_list()))
    split_values = shard_df["split"].unique().to_list()
    if len(split_values) != 1:
        raise ValueError(f"Output shard {rel_path} contains multiple splits: {split_values}")

    context_symbols = [vocab.context_symbols[idx] for idx in context_codes]
    return {
        "shard_path": rel_path,
        "source": source,
        "size": shard_df.height,
        "scalar_values": int(shard_df["n_values"].sum()),
        "split": split_values[0],
        "context": context_symbols[0] if len(context_symbols) == 1 else "mixed",
        "datasets": [vocab.dataset_symbols[idx] for idx in dataset_codes],
        "contexts": context_symbols,
        "perturbations": [vocab.perturb_symbols[idx] for idx in perturb_codes],
        "readouts": [vocab.readout_symbols[idx] for idx in readout_codes],
    }


def write_source_manifest(output_root: Path, stats: SourceStats) -> None:
    split_stats = {
        split: {
            "raw_scalar_values": values.raw_scalar_values,
            "joined_scalar_values": values.joined_scalar_values,
            "dropped_without_split": values.dropped_without_split,
            "output_samples": values.output_samples,
            "output_shards": values.output_shards,
        }
        for split, values in stats.split_stats.items()
    }
    payload = {
        "source": stats.source,
        "raw_scalar_values": stats.raw_scalar_values,
        "joined_scalar_values": stats.joined_scalar_values,
        "dropped_without_split": stats.dropped_without_split,
        "scalar_values_written": stats.scalar_values_written,
        "output_samples": stats.output_samples,
        "output_shards": stats.output_shards,
        "split_stats": split_stats,
    }
    source_dir = output_root / stats.source
    atomic_write_text(source_dir / "source_manifest.yaml", yaml.safe_dump(payload, sort_keys=False))
    atomic_write_text(source_dir / "source_manifest.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")


def expected_scalar_values_from_plibdata(plibdata_root: Path, source: str) -> int | None:
    metadata_path = plibdata_root / source / "metadata.parquet"
    if not metadata_path.is_file():
        return None
    return int(pl.read_parquet(metadata_path, columns=["size"])["size"].sum())


def source_lock_path(output_root: Path, source: str) -> Path:
    safe_source = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in source)
    lock_dir = output_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / f"{safe_source}.lock"


def process_source(
    spec: SourceSpec,
    output_root: Path,
    vocab: Vocab,
    target_values_per_shard: int,
    compression: str,
    overwrite: bool,
    skip_existing: bool,
    include_symbols: bool,
    value_dtype_name: str,
    feature_dtype_name: str,
) -> SourceStats:
    with open(source_lock_path(output_root, spec.source), "w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        return _process_source_locked(
            spec=spec,
            output_root=output_root,
            vocab=vocab,
            target_values_per_shard=target_values_per_shard,
            compression=compression,
            overwrite=overwrite,
            skip_existing=skip_existing,
            include_symbols=include_symbols,
            value_dtype_name=value_dtype_name,
            feature_dtype_name=feature_dtype_name,
        )


def _process_source_locked(
    spec: SourceSpec,
    output_root: Path,
    vocab: Vocab,
    target_values_per_shard: int,
    compression: str,
    overwrite: bool,
    skip_existing: bool,
    include_symbols: bool,
    value_dtype_name: str,
    feature_dtype_name: str,
) -> SourceStats:
    source_dir = output_root / spec.source
    done_file = source_dir / "source_manifest.yaml"
    if skip_existing and done_file.exists() and not overwrite:
        log(f"Skipping {spec.source}: {done_file} already exists.")
        return SourceStats(source=spec.source)
    if overwrite and source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)

    value_dtype = pl.Float32 if value_dtype_name == "float32" else pl.Float64
    feature_dtype = pl.Float32 if feature_dtype_name == "float32" else pl.Float64

    split_lookup = (
        pl.read_parquet(spec.annotations_path, columns=JOIN_KEYS + ["split"])
        .unique(subset=JOIN_KEYS, maintain_order=True)
        .filter(pl.col("context") == spec.context)
    )

    stats = SourceStats(source=spec.source)
    metadata_rows: list[dict[str, Any]] = []
    output_shard_index = 0
    buffer: list[pl.DataFrame] = []
    buffered_values = 0

    def flush(force: bool = False) -> None:
        nonlocal output_shard_index, buffer, buffered_values
        if not buffer:
            return
        if not force and buffered_values < target_values_per_shard:
            return
        shard_df = pl.concat(buffer, how="vertical", rechunk=True)
        shard_name = f"shard_{output_shard_index:06d}.parquet"
        rel_path = f"{spec.source}/{shard_name}"
        atomic_write_parquet(shard_df, output_root / rel_path, compression)
        metadata_rows.append(metadata_row(spec.source, rel_path, shard_df, vocab))

        scalar_values = int(shard_df["n_values"].sum())
        split = str(shard_df["split"][0])
        stats.output_shards += 1
        stats.output_samples += shard_df.height
        stats.scalar_values_written += scalar_values
        split_stats = stats.split_stats.setdefault(split, SplitStats())
        split_stats.output_shards += 1
        split_stats.output_samples += shard_df.height
        output_shard_index += 1
        buffer = []
        buffered_values = 0

    log(
        f"{spec.source}: cache_root={spec.cache_root}, raw folder={spec.dataset_folder}, "
        f"context={spec.context}, files={len(spec.files)}"
    )
    for raw_path in spec.files:
        raw = pl.read_parquet(raw_path, columns=RAW_COLUMNS).filter(pl.col("context") == spec.context)
        if raw.is_empty():
            continue
        n_raw = raw.height
        joined = raw.join(split_lookup, on=JOIN_KEYS, how="inner")
        n_joined = joined.height
        n_dropped = n_raw - n_joined

        stats.raw_scalar_values += n_raw
        stats.joined_scalar_values += n_joined
        stats.dropped_without_split += n_dropped
        if n_dropped:
            log(f"{spec.source}: {raw_path.name}: dropped {n_dropped} rows with no split assignment")
        if joined.is_empty():
            continue

        for split in SPLIT_ORDER:
            split_df = joined.filter(pl.col("split") == split)
            if split_df.is_empty():
                continue
            split_stats = stats.split_stats.setdefault(split, SplitStats())
            split_stats.raw_scalar_values += split_df.height
            split_stats.joined_scalar_values += split_df.height
            grouped = group_to_multiout(split_df, split, vocab, value_dtype, feature_dtype, include_symbols)
            if grouped.is_empty():
                continue
            if buffer and str(buffer[0]["split"][0]) != split:
                flush(force=True)
            buffer.append(grouped)
            buffered_values += int(grouped["n_values"].sum())
            flush(force=False)

    flush(force=True)

    if stats.scalar_values_written != stats.joined_scalar_values:
        raise RuntimeError(
            f"{spec.source}: wrote {stats.scalar_values_written} scalar values but joined "
            f"{stats.joined_scalar_values} raw rows to split labels."
        )

    expected = expected_scalar_values_from_plibdata(spec.plibdata_root, spec.source)
    if expected is not None and expected != stats.scalar_values_written:
        raise RuntimeError(
            f"{spec.source}: multiout wrote {stats.scalar_values_written} scalar values, "
            f"but existing notebook plibdata metadata has {expected}."
        )

    metadata_df = pl.DataFrame(metadata_rows)
    atomic_write_parquet(metadata_df, source_dir / "metadata.parquet", compression)
    write_source_manifest(output_root, stats)
    log(
        f"{spec.source}: done, samples={stats.output_samples}, "
        f"scalar_values={stats.scalar_values_written}, shards={stats.output_shards}"
    )
    return stats


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--cache-root", type=Path, default=None, help="Cache root containing raw_datasets/ and annotations/."
    )
    parser.add_argument("--raw-root", type=Path, default=None, help="Override raw_datasets root.")
    parser.add_argument("--annotations", type=Path, default=None, help="Override df_annot_split.parquet path.")
    parser.add_argument(
        "--reference-plibdata-root",
        type=Path,
        default=None,
        help="Optional original scalar plibdata root used only for count validation.",
    )
    parser.add_argument(
        "--fallback-cache-root",
        action="append",
        type=Path,
        default=None,
        help="Additional cache root(s) to search when a source is missing from the primary cache.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--source", action="append", default=None, help="Configured source name to process. Repeatable."
    )
    parser.add_argument("--source-index", type=int, default=None, help="Process the Nth configured source.")
    parser.add_argument("--max-sources", type=int, default=None, help="For smoke tests, process only first N sources.")
    parser.add_argument("--target-values-per-shard", type=int, default=8_000_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--value-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--float-feature-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--include-symbols", action="store_true")
    parser.add_argument(
        "--vocab-from-selected-sources",
        action="store_true",
        help="Smoke-test helper; do not use for full training cache.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    plibdata_root, all_sources, config = load_config(args.config)
    config_annotation = config.get("data_configs", [{}])[0].get("on_disk_split_annotation_path")
    annotation_overridden = args.annotations is not None or bool(config_annotation)
    if args.annotations is None and config_annotation:
        args.annotations = Path(config_annotation)
    cache_root, raw_root, annotations_path = resolve_cache_paths(args, plibdata_root)
    reference_plibdata_root = args.reference_plibdata_root or plibdata_root
    cache_paths = [
        CachePaths(
            cache_root=cache_root,
            raw_root=raw_root,
            annotations_path=annotations_path,
            plibdata_root=reference_plibdata_root,
        )
    ]
    for fallback_cache_root in args.fallback_cache_root or []:
        fallback_raw_root = fallback_cache_root / "raw_datasets"
        fallback_annotations_path = annotations_path if annotation_overridden else fallback_cache_root / "annotations" / "df_annot_split.parquet"
        fallback_plibdata_root = fallback_cache_root / "plibdata"
        if not fallback_raw_root.is_dir():
            raise FileNotFoundError(f"Missing fallback raw dataset root: {fallback_raw_root}")
        if not fallback_annotations_path.is_file():
            raise FileNotFoundError(f"Missing fallback split annotation file: {fallback_annotations_path}")
        cache_paths.append(
            CachePaths(
                cache_root=fallback_cache_root,
                raw_root=fallback_raw_root,
                annotations_path=fallback_annotations_path,
                plibdata_root=fallback_plibdata_root,
            )
        )
    all_specs = make_source_specs(cache_paths, all_sources)

    selected_specs = list(all_specs)
    if args.max_sources is not None:
        selected_specs = selected_specs[: args.max_sources]
    if args.source:
        spec_by_source = {spec.source: spec for spec in all_specs}
        unknown = sorted(set(args.source) - set(spec_by_source))
        if unknown:
            raise ValueError(f"Unknown source(s): {unknown}")
        selected_specs = [spec_by_source[source] for source in args.source]
    if args.source_index is not None:
        if args.source_index < 0 or args.source_index >= len(all_specs):
            raise ValueError(f"--source-index {args.source_index} outside [0, {len(all_specs) - 1}]")
        selected_specs = [all_specs[args.source_index]]

    log(f"config={args.config}")
    log(f"cache_root={cache_root}")
    log(f"raw_root={raw_root}")
    log(f"annotations={annotations_path}")
    log(f"reference_plibdata_root={reference_plibdata_root}")
    for fallback_paths in cache_paths[1:]:
        log(f"fallback_cache_root={fallback_paths.cache_root}")
    log(f"output_root={args.output_root}")
    log(f"selected_sources={len(selected_specs)} of {len(all_specs)}")
    if args.dry_run:
        for spec in selected_specs:
            log(
                f"dry-run source: {spec.source} <- {spec.dataset_folder}, context={spec.context}, "
                f"files={len(spec.files)}, cache_root={spec.cache_root}"
            )
        return 0

    start = time.time()
    vocab_specs = selected_specs if args.vocab_from_selected_sources else all_specs
    vocab = get_or_build_vocab(args.output_root, vocab_specs, args.compression)
    write_dataset_scaffolding(
        output_root=args.output_root,
        config_path=args.config,
        config=config,
        cache_paths=cache_paths,
        source_specs=all_specs,
        sources=all_sources,
        target_values_per_shard=args.target_values_per_shard,
        value_dtype=args.value_dtype,
        float_feature_dtype=args.float_feature_dtype,
        include_symbols=args.include_symbols,
    )

    total = SourceStats(source="TOTAL")
    for spec in selected_specs:
        stats = process_source(
            spec=spec,
            output_root=args.output_root,
            vocab=vocab,
            target_values_per_shard=args.target_values_per_shard,
            compression=args.compression,
            overwrite=args.overwrite,
            skip_existing=args.skip_existing,
            include_symbols=args.include_symbols,
            value_dtype_name=args.value_dtype,
            feature_dtype_name=args.float_feature_dtype,
        )
        total.raw_scalar_values += stats.raw_scalar_values
        total.joined_scalar_values += stats.joined_scalar_values
        total.dropped_without_split += stats.dropped_without_split
        total.scalar_values_written += stats.scalar_values_written
        total.output_samples += stats.output_samples
        total.output_shards += stats.output_shards

    elapsed = (time.time() - start) / 60.0
    log(
        f"finished in {elapsed:.2f} min; samples={total.output_samples}, "
        f"scalar_values={total.scalar_values_written}, shards={total.output_shards}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

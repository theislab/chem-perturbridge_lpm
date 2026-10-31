#!/usr/bin/env python3
"""Write plibdata shards for the L1000-compound-narrowed "all" split.

Mirrors notebooks/03_LPM_style_write_shards.ipynb, but reads the narrowed
annotation under .plib_cache_paper_narrowed and writes shards there.

Default paths (relative to repo root):
  raw:    .plib_cache_paper/raw_datasets
  annot:  .plib_cache_paper_narrowed/annotations/df_annot_split_narrowed.parquet
  shards: .plib_cache_paper_narrowed/plibdata

Usage:
  python scripts/write_narrowed_plibdata_shards.py
  python scripts/write_narrowed_plibdata_shards.py --build-annot
  sbatch scripts/run_write_narrowed_plibdata_shards.sh
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / ".plib_cache_paper" / "raw_datasets"
DEFAULT_ANNOT = (
    REPO_ROOT / ".plib_cache_paper_narrowed" / "annotations" / "df_annot_split_narrowed.parquet"
)
DEFAULT_SOURCE_ANNOT = REPO_ROOT / ".plib_cache_paper" / "annotations" / "df_annot_split.parquet"
DEFAULT_SHARDS = REPO_ROOT / ".plib_cache_paper_narrowed" / "plibdata"
DEFAULT_SHARDSIZE = 200_000

JOIN_KEYS = ["dataset", "context", "perturbation"]
SHARD_COLS = ["dataset", "context", "perturbation", "readout", "log_dose", "time", "value", "split"]

# raw_datasets folder names that are not part of the non-OP3 paper split
SKIP_RAW_DATASETS = {"op3"}


def build_narrowed_annot(source_annot: Path, out_annot: Path) -> pd.DataFrame:
    """Rebuild narrowed annotation from paper df_annot_split (same rules as agreed)."""
    df = pd.read_parquet(source_annot)
    df["perturbation"] = df["perturbation"].astype(str)
    df["context"] = df["context"].astype(str)

    lincs = df["dataset"].str.contains("LINCS", regex=False)
    l1000_train = set(df.loc[lincs & (df["split"] == "train"), "perturbation"])
    l1000_val = set(df.loc[lincs & (df["split"] == "val"), "perturbation"])
    l1000_test = set(df.loc[lincs & (df["split"] == "test"), "perturbation"])

    train = df[(df["split"] == "train") & df["perturbation"].isin(l1000_train)].copy()
    train_ctxs = set(train["context"])

    val = df[
        (df["split"] == "val")
        & df["perturbation"].isin(l1000_val)
        & df["context"].isin(train_ctxs)
    ].copy()
    test = df[
        (df["split"] == "test")
        & df["perturbation"].isin(l1000_test)
        & df["context"].isin(train_ctxs)
    ].copy()

    final = pd.concat([train, val, test], ignore_index=True)
    final = final.sort_values(["dataset", "split", "context", "perturbation"]).reset_index(drop=True)

    out_annot.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(out_annot, index=False)

    provenance = {
        "rules": {
            "train": "split==train AND perturbation in L1000_train",
            "val": "split==val AND perturbation in L1000_val AND context in train_after_filtration",
            "test": "split==test AND perturbation in L1000_test AND context in train_after_filtration",
            "note": (
                "10 val + 15 test L1000-holdout-only compounds kept even if absent "
                "from filtered train (to match L1000 compound counts)"
            ),
        },
        "counts": {
            "train_pairs": int((final["split"] == "train").sum()),
            "val_pairs": int((final["split"] == "val").sum()),
            "test_pairs": int((final["split"] == "test").sum()),
            "train_compounds": int(final.loc[final["split"] == "train", "perturbation"].nunique()),
            "val_compounds": int(final.loc[final["split"] == "val", "perturbation"].nunique()),
            "test_compounds": int(final.loc[final["split"] == "test", "perturbation"].nunique()),
        },
        "l1000_val_only_compounds_kept_without_train": sorted(l1000_val - l1000_train),
        "l1000_test_only_compounds_kept_without_train": sorted(l1000_test - l1000_train),
    }
    (out_annot.parent / "narrowing_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"[annot] wrote {out_annot} ({len(final)} pairs)")
    print(f"[annot] split counts:\n{final['split'].value_counts().to_string()}")
    return final


def _append_context_chunk(
    ctx_df: pd.DataFrame,
    out_dir: Path,
    shardsize: int,
    next_shard_id: int,
) -> tuple[int, list[dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    long = ctx_df[SHARD_COLS].copy()
    long["perturbation"] = long["perturbation"].astype(str)
    long = long.sort_values("split", kind="mergesort").reset_index(drop=True)
    long["shard_id_per_split"] = long.groupby("split").cumcount() // shardsize
    combo = long["split"].astype(str) + "::" + long["shard_id_per_split"].astype(str)
    long["shard_id"] = pd.factorize(combo)[0] + next_shard_id
    long = long.drop(columns="shard_id_per_split")

    metadata_rows: list[dict] = []
    max_id = next_shard_id - 1
    for shard_id, shard_df in long.groupby("shard_id", sort=True):
        shard_name = f"shard_{int(shard_id):06d}.parquet"
        shard_df.drop(columns=["split", "shard_id"]).reset_index(drop=True).to_parquet(
            out_dir / shard_name, engine="pyarrow", index=False
        )
        metadata_rows.append(
            {
                "shard_path": f"{out_dir.name}/{shard_name}",
                "size": int(len(shard_df)),
                "split": str(shard_df["split"].iloc[0]),
                "context": str(shard_df["context"].iloc[0]),
                "datasets": str(shard_df["dataset"].iloc[0]),
                "perturbations": sorted({p for s in shard_df["perturbation"] for p in s.split("+")}),
                "log_doses": sorted(shard_df["log_dose"].dropna().unique().tolist()),
                "times": sorted(shard_df["time"].dropna().unique().tolist()),
                "readouts": shard_df["readout"].unique().tolist(),
            }
        )
        max_id = max(max_id, int(shard_id))

    return max_id + 1, metadata_rows


def _reset_context_folder(out_dir: Path, shardsize: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.iterdir():
        (shutil.rmtree if f.is_dir() else Path.unlink)(f)
    (out_dir / "info.json").write_text(
        json.dumps(
            {
                "PDATA_FORMAT_VERSION": 1,
                "CONTEXT_MODULE_HASH": "",
                "SHARDSIZE": shardsize,
            }
        )
    )


def _finalize_context_folder(out_dir: Path, metadata_rows: list[dict]) -> None:
    pd.DataFrame(metadata_rows).to_parquet(out_dir / "metadata.parquet", engine="pyarrow", index=False)


def write_shards(
    raw_root: Path,
    annot_path: Path,
    shards_root: Path,
    shardsize: int,
    skip_datasets: set[str],
) -> None:
    df_annot_split = pd.read_parquet(annot_path)
    for col in JOIN_KEYS:
        df_annot_split[col] = df_annot_split[col].astype(str)

    split_lookup = df_annot_split[JOIN_KEYS + ["split"]].drop_duplicates(JOIN_KEYS)

    shards_root.mkdir(parents=True, exist_ok=True)
    print(f"[shards] writing under: {shards_root}")
    print(f"[shards] raw datasets from: {raw_root}")
    print(f"[shards] annotation: {annot_path} ({len(df_annot_split)} pairs)")

    source_folders = sorted(
        p
        for p in os.listdir(raw_root)
        if (raw_root / p).is_dir() and p not in skip_datasets
    )
    print(f"[shards] source folders ({len(source_folders)}): {source_folders}")

    shard_counters: dict[str, int] = defaultdict(int)
    metadata_per_folder: dict[str, list[dict]] = defaultdict(list)
    folder_initialized: set[str] = set()

    for dataset_folder in tqdm(source_folders, desc="datasets"):
        dataset_dir = raw_root / dataset_folder
        files = sorted(f for f in os.listdir(dataset_dir) if f.endswith(".parquet"))
        if not files:
            print(f"  [skip] {dataset_folder}: no parquet files")
            continue

        touched_in_this_dataset: set[str] = set()

        for file in tqdm(files, desc=dataset_folder, leave=False):
            df = pd.read_parquet(dataset_dir / file)
            if df.empty:
                continue

            for col in JOIN_KEYS:
                df[col] = df[col].astype(str)

            n_before = len(df)
            df = df.merge(split_lookup, on=JOIN_KEYS, how="inner")
            n_lost = n_before - len(df)
            if n_lost:
                print(
                    f"  [info] {dataset_folder}/{file}: dropped {n_lost}/{n_before} "
                    "rows not in narrowed split"
                )
            if df.empty:
                continue

            for context, ctx_df in df.groupby("context", sort=True):
                folder_name = f"{dataset_folder}_{context}"
                out_dir = shards_root / folder_name

                if folder_name not in folder_initialized:
                    _reset_context_folder(out_dir, shardsize)
                    folder_initialized.add(folder_name)

                next_id, meta = _append_context_chunk(
                    ctx_df, out_dir, shardsize, shard_counters[folder_name]
                )
                shard_counters[folder_name] = next_id
                metadata_per_folder[folder_name].extend(meta)
                touched_in_this_dataset.add(folder_name)

        for folder_name in touched_in_this_dataset:
            _finalize_context_folder(shards_root / folder_name, metadata_per_folder[folder_name])

    total_shards = sum(len(v) for v in metadata_per_folder.values())
    total_folders = len(metadata_per_folder)
    print(
        f"[shards] done: wrote {total_shards} shards across {total_folders} "
        f"dataset_context folders -> {shards_root}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    p.add_argument("--annot", type=Path, default=DEFAULT_ANNOT)
    p.add_argument("--source-annot", type=Path, default=DEFAULT_SOURCE_ANNOT)
    p.add_argument("--shards-root", type=Path, default=DEFAULT_SHARDS)
    p.add_argument("--shardsize", type=int, default=DEFAULT_SHARDSIZE)
    p.add_argument(
        "--build-annot",
        action="store_true",
        help="Rebuild narrowed annotation from --source-annot before sharding.",
    )
    p.add_argument(
        "--skip-dataset",
        action="append",
        default=[],
        help="Raw dataset folder name to skip (repeatable). op3 is skipped by default.",
    )
    p.add_argument(
        "--include-op3",
        action="store_true",
        help="Do not skip the op3 raw_datasets folder (still only rows in annot are kept).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    skip = set(SKIP_RAW_DATASETS)
    if args.include_op3:
        skip.discard("op3")
    skip.update(args.skip_dataset)

    if args.build_annot or not args.annot.exists():
        if not args.source_annot.exists():
            print(f"ERROR: source annot not found: {args.source_annot}", file=sys.stderr)
            return 1
        if not args.annot.exists():
            print(f"[annot] {args.annot} missing — rebuilding")
        build_narrowed_annot(args.source_annot, args.annot)

    if not args.raw_root.is_dir():
        print(f"ERROR: raw root not found: {args.raw_root}", file=sys.stderr)
        return 1

    write_shards(
        raw_root=args.raw_root,
        annot_path=args.annot,
        shards_root=args.shards_root,
        shardsize=args.shardsize,
        skip_datasets=skip,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

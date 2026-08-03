#!/usr/bin/env python3
"""Score Scratch vs FT Morgan learned fixmol → unit-level test RMSE.

Writes per-(dataset, seed, molecule, context, log_dose, time) RMSE for Chem-style
nested BCa in ``notebooks_work/paper10_bootstrap.ipynb``.

Example:
  CUDA_VISIBLE_DEVICES=0 lpm_training_venv/bin/python scripts/paper10_compound_win_rates.py
  CUDA_VISIBLE_DEVICES=0 lpm_training_venv/bin/python scripts/paper10_compound_win_rates.py --check-only
  sbatch scripts/paper10_unit_rmse_dose_time.sbatch
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Heavy scoring deps (torch / scanpy / numba) are imported lazily in
# `_load_scoring_deps()` so --write-task-list / --merge-parts stay lightweight.

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
LONG_TSV = RESULTS / "lpm_paper10_results_current_check_long.tsv"
DEFAULT_SHARD_ROOT = Path(
    "/lustre/groups/ml01/workspace/olga.novitskaia/lpm_style/.plib_cache/plibdata_multiout/"
    "lpm_multiout_all_data_plus_tahoe_novartis_op3_molholdout_h100_2x_bs4096_200epoch_lustre"
)
ARTUR_RESULTS = Path(
    "/lustre/groups/ml01/workspace/artur.szalata/code/chem-perturbridge_lpm/.plib_cache/results"
)

REF_FAMILY = "scratch_target_only"
VAR_FAMILY = "finetune_morgan_learned_fixed_updated_embeddings"

DATASETS: dict[str, tuple[str, ...]] = {
    "lincs_phase1": ("l1000_phase1_",),
    "lincs_phase2": ("l1000_phase2_",),
    "novartis": ("novartis_",),
    "vcpi_0001": ("vcpi_0001_",),
    "op3": ("op3_",),
    "tahoe100": ("tahoe_",),
    "cigs_tcm": ("cigs_tcm_",),
    "dilimap_train": ("dili_train_",),
    "gdpx2": ("gdpx2_",),
    "sciplex": ("sciplex_",),
    "cigs_mce": ("cigs_mce_",),
}
DISPLAY = {
    "lincs_phase1": "LINCS Phase I",
    "lincs_phase2": "LINCS Phase II",
    "novartis": "Novartis DRUG-seq",
    "vcpi_0001": "VCPI vcpi-0001",
    "op3": "OP3",
    "tahoe100": "Tahoe-100M",
    "cigs_tcm": "CIGS TCM",
    "dilimap_train": "DILImap train",
    "gdpx2": "Ginkgo GDPx2",
    "sciplex": "sci-Plex",
    "cigs_mce": "CIGS MCE",
}


def _load_scoring_deps():
    import polars as pl
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compute_null_baselines import MetricAccumulator  # noqa: E402
    from evaluate_checkpoint_by_dataset import (  # noqa: E402
        CheckpointModel,
        batch_from_frame,
        iter_selected_shards,
    )

    return pl, torch, MetricAccumulator, CheckpointModel, batch_from_frame, iter_selected_shards


def remap_ckpt(path: str | Path) -> Path:
    p = Path(str(path).replace("/ictstr01/groups/", "/lustre/groups/"))
    if not p.exists() and "chem-perturbridge_lpm/.plib_cache/results/" in str(p):
        name = str(p).split("chem-perturbridge_lpm/.plib_cache/results/")[-1]
        alt = ARTUR_RESULTS / name
        if alt.exists():
            return alt
    return p


def score_checkpoint_unit_rmse(
    checkpoint: Path,
    shard_root: Path,
    prefixes: tuple[str, ...],
    code_to_molecule: dict[int, str],
    code_to_context: dict[int, str],
    *,
    device,
    row_chunk: int = 256,
    deps=None,
) -> tuple[list[dict], float]:
    """Accumulate RMSE by (molecule, context, log_dose, time); return unit rows + pooled RMSE."""
    if deps is None:
        deps = _load_scoring_deps()
    _pl, torch, MetricAccumulator, CheckpointModel, batch_from_frame, iter_selected_shards = deps
    model = CheckpointModel(checkpoint, device)
    per_unit: dict[tuple[int, int, float, float], MetricAccumulator] = defaultdict(MetricAccumulator)
    pooled = MetricAccumulator()
    with torch.no_grad():
        for _path, frame in iter_selected_shards(shard_root, "test", prefixes, None):
            for lo in range(0, len(frame), row_chunk):
                hi = min(lo + row_chunk, len(frame))
                chunk = frame[lo:hi]
                batch = batch_from_frame(frame, lo, hi, device)
                prediction = model.forward(batch)
                counts_np = chunk["n_values"].to_numpy().astype("int64")
                counts = torch.tensor(counts_np)
                genes = torch.tensor(
                    np.concatenate(chunk["readout_codes"].to_list()).astype("int64")
                ).to(device)
                rows = torch.repeat_interleave(torch.arange(hi - lo, dtype=torch.long), counts).to(device)
                selected_t = prediction[rows, genes]
                if model.offset_table is not None:
                    selected_t = selected_t + model.offset_for(batch, rows, genes)
                selected = selected_t.cpu().numpy().astype(np.float64)
                truth = np.concatenate(chunk["values"].to_list()).astype(np.float64)
                mol_row = np.asarray(chunk["perturbation_codes"].list.first().to_list(), dtype=np.int64)
                ctx_row = chunk["context_code"].to_numpy().astype(np.int64)
                dose_row = chunk["log_dose"].to_numpy().astype(np.float64)
                time_row = chunk["time"].to_numpy().astype(np.float64)
                pooled.add(truth, selected)
                # Each shard row is one (mol, context, dose, time) condition.
                offset = 0
                for i, n in enumerate(counts_np):
                    n = int(n)
                    key = (int(mol_row[i]), int(ctx_row[i]), float(dose_row[i]), float(time_row[i]))
                    per_unit[key].add(truth[offset : offset + n], selected[offset : offset + n])
                    offset += n

    unit_rows = []
    for (mol_c, ctx_c, log_dose, time_h), acc in per_unit.items():
        if mol_c not in code_to_molecule or ctx_c not in code_to_context:
            continue
        unit_rows.append(
            {
                "molecule": code_to_molecule[mol_c],
                "context": code_to_context[ctx_c],
                "log_dose": float(log_dose),
                "time": float(time_h),
                "rmse": float(acc.finalize()["RMSE"]),
            }
        )
    return unit_rows, float(pooled.finalize()["RMSE"])


def _part_name(dataset_slug: str, family: str, seed: int) -> str:
    fam_short = "scratch" if family == REF_FAMILY else "ft"
    return f"{dataset_slug}__{fam_short}__seed{int(seed)}.tsv"


def build_task_frame(long: pd.DataFrame, datasets: list[str], families: list[str], max_seeds: int | None) -> pd.DataFrame:
    rows = []
    for slug in datasets:
        sub = long[long["dataset_slug"] == slug]
        seeds = sorted(int(s) for s in sub["seed"].unique())
        if max_seeds is not None:
            seeds = seeds[:max_seeds]
        for family in families:
            for seed in seeds:
                row = sub[(sub["model_family"] == family) & (sub["seed"] == seed)]
                if row.empty:
                    continue
                rows.append(
                    {
                        "dataset_slug": slug,
                        "model_family": family,
                        "seed": int(seed),
                        "checkpoint_path": str(row.iloc[0]["checkpoint_path"]),
                        "tsv_test_rmse": float(row.iloc[0]["test_rmse"]),
                        "part_name": _part_name(slug, family, seed),
                    }
                )
    tasks = pd.DataFrame(rows)
    if not tasks.empty:
        tasks = tasks.reset_index(drop=True)
        tasks.insert(0, "task_index", tasks.index.astype(int))
    return tasks


def merge_parts(parts_dir: Path, output_path: Path) -> int:
    files = sorted(parts_dir.glob("*.tsv"))
    if not files:
        print(f"No part TSVs in {parts_dir}")
        return 1
    frames = [pd.read_csv(p, sep="\t") for p in files]
    scores = pd.concat(frames, ignore_index=True)
    # Prefer one row set per (dataset, family, seed, molecule, context, log_dose, time)
    key = ["dataset_slug", "model_family", "seed", "molecule", "context", "log_dose", "time"]
    before = len(scores)
    scores = scores.drop_duplicates(subset=key, keep="last")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, sep="\t", index=False)
    print(f"Merged {len(files)} parts → {output_path}  rows={len(scores)} (from {before})")
    print(scores.groupby(["dataset_slug", "model_family"]).size().to_string())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--long-tsv", type=Path, default=LONG_TSV)
    parser.add_argument("--shard-root", type=Path, default=DEFAULT_SHARD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "paper10_bootstrap")
    parser.add_argument(
        "--output-name",
        default="paper10_unit_rmse_long_scratch_vs_ft_morgan_learned_fixmol_mol_ctx_dose_time.tsv",
        help="Unit RMSE long table filename under --output-dir",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument(
        "--families",
        nargs="+",
        default=[REF_FAMILY, VAR_FAMILY],
        choices=[REF_FAMILY, VAR_FAMILY],
    )
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional explicit seed list")
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=None,
        help="Write one TSV per (dataset, family, seed) here (for array jobs)",
    )
    parser.add_argument(
        "--task-list",
        type=Path,
        default=None,
        help="Task TSV with columns task_index,dataset_slug,model_family,seed,...",
    )
    parser.add_argument("--write-task-list", action="store_true", help="Write --task-list and exit")
    parser.add_argument("--task-index", type=int, default=None, help="Score only this task_index from --task-list")
    parser.add_argument("--merge-parts", action="store_true", help="Merge --parts-dir into --output-name and exit")
    parser.add_argument("--skip-existing-parts", action="store_true", help="Skip tasks whose part TSV already exists")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.parts_dir or (args.output_dir / "unit_rmse_parts")
    task_list_path = args.task_list or (args.output_dir / "unit_rmse_score_tasks.tsv")

    if args.merge_parts:
        return merge_parts(parts_dir, args.output_dir / args.output_name)

    print(f"Reading {args.long_tsv} ...", flush=True)
    long = pd.read_csv(
        args.long_tsv,
        sep="\t",
        usecols=["model_family", "seed", "dataset_slug", "checkpoint_path", "test_rmse", "status"],
    )
    long = long[
        long["model_family"].isin(args.families)
        & long["dataset_slug"].isin(args.datasets)
        & (long["status"] == "ok")
    ].copy()
    long["checkpoint_path"] = long["checkpoint_path"].map(remap_ckpt)
    if args.seeds is not None:
        long = long[long["seed"].isin(args.seeds)].copy()
    print(f"Filtered long TSV rows: {len(long)}", flush=True)

    tasks = build_task_frame(long, list(args.datasets), list(args.families), args.max_seeds)
    if args.write_task_list:
        task_list_path.parent.mkdir(parents=True, exist_ok=True)
        tasks.to_csv(task_list_path, sep="\t", index=False)
        print(f"Wrote {len(tasks)} tasks → {task_list_path}", flush=True)
        return 0 if len(tasks) else 1

    access_rows = []
    for _, r in long.iterrows():
        p = Path(r["checkpoint_path"])
        status = "missing"
        if p.exists():
            try:
                with open(p, "rb") as f:
                    f.read(1)
                status = "ok"
            except PermissionError:
                status = "permission_denied"
        access_rows.append(
            {
                "model_family": r["model_family"],
                "dataset_slug": r["dataset_slug"],
                "seed": int(r["seed"]),
                "checkpoint_path": str(p),
                "status": status,
                "tsv_test_rmse": float(r["test_rmse"]),
            }
        )
    access = pd.DataFrame(access_rows)
    access_path = args.output_dir / "paper10_checkpoint_access_check.tsv"
    access.to_csv(access_path, sep="\t", index=False)
    print(access["status"].value_counts().to_string())
    print(f"Saved {access_path}")
    if int((access["status"] == "ok").sum()) == 0:
        print("No readable checkpoints.")
        return 2
    if args.check_only:
        return 0

    if args.task_index is not None:
        if not task_list_path.is_file():
            raise FileNotFoundError(f"Missing task list {task_list_path}; run with --write-task-list first")
        tasks = pd.read_csv(task_list_path, sep="\t")
        tasks = tasks[tasks["task_index"] == int(args.task_index)].copy()
        if tasks.empty:
            raise ValueError(f"task_index={args.task_index} not in {task_list_path}")

    if not args.shard_root.exists():
        raise FileNotFoundError(args.shard_root)
    deps = _load_scoring_deps()
    pl, torch, *_rest = deps
    perturb_vocab = pl.read_parquet(args.shard_root / "vocab" / "perturb_vocab.parquet")
    context_vocab = pl.read_parquet(args.shard_root / "vocab" / "context_vocab.parquet")
    code_to_molecule = {
        int(c): str(s) for c, s in zip(perturb_vocab["code"].to_list(), perturb_vocab["symbol"].to_list())
    }
    code_to_context = {
        int(c): str(s) for c, s in zip(context_vocab["code"].to_list(), context_vocab["symbol"].to_list())
    }

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device={device}  shard_root={args.shard_root}")
    parts_dir.mkdir(parents=True, exist_ok=True)

    score_rows: list[dict] = []
    write_parts = args.task_index is not None or args.parts_dir is not None
    for _, task in tasks.iterrows():
        slug = str(task["dataset_slug"])
        family = str(task["model_family"])
        seed = int(task["seed"])
        part_path = parts_dir / str(task.get("part_name", _part_name(slug, family, seed)))
        if args.skip_existing_parts and part_path.is_file() and part_path.stat().st_size > 0:
            print(f"[skip existing] {part_path.name}", flush=True)
            continue
        status_rows = access.loc[
            (access.model_family == family) & (access.dataset_slug == slug) & (access.seed == seed),
            "status",
        ]
        if status_rows.empty or status_rows.iloc[0] != "ok":
            print(f"[skip not-ok] {family} {slug} seed={seed}", flush=True)
            continue
        ckpt = Path(task["checkpoint_path"])
        tsv_rmse = float(task["tsv_test_rmse"])
        t0 = time.time()
        print(f"Scoring {family} {slug} seed={seed} ...", flush=True)
        units, pooled = score_checkpoint_unit_rmse(
            ckpt,
            args.shard_root,
            DATASETS[slug],
            code_to_molecule,
            code_to_context,
            device=device,
            row_chunk=args.row_chunk,
            deps=deps,
        )
        dt = time.time() - t0
        print(
            f"  pooled={pooled:.6f} tsv={tsv_rmse:.6f} |diff|={abs(pooled - tsv_rmse):.3e}  "
            f"n_units={len(units)}  ({dt:.0f}s)",
            flush=True,
        )
        rows = [
            {
                "dataset_slug": slug,
                "dataset": DISPLAY[slug],
                "model_family": family,
                "seed": int(seed),
                "molecule": u["molecule"],
                "context": u["context"],
                "log_dose": u["log_dose"],
                "time": u["time"],
                "rmse": u["rmse"],
                "pooled_rmse": pooled,
                "tsv_test_rmse": tsv_rmse,
            }
            for u in units
        ]
        if write_parts:
            pd.DataFrame(rows).to_csv(part_path, sep="\t", index=False)
            print(f"  saved {part_path}", flush=True)
        else:
            score_rows.extend(rows)

    if write_parts:
        print(f"Part(s) written under {parts_dir}")
        return 0

    if not score_rows:
        print("No scores written.")
        return 1

    scores = pd.DataFrame(score_rows)
    out = args.output_dir / args.output_name
    scores.to_csv(out, sep="\t", index=False)
    print(f"Saved {out}  rows={len(scores)}")
    print(scores.groupby(["dataset_slug", "model_family"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

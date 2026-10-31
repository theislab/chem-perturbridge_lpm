"""Groupby null baselines for the multiout LPM task.

Fits trivial group-mean predictors on the train split of an on-disk multiout shard
root and scores them on val/test with the same metrics perturb-lib uses
(``sklearn`` RMSE / MAE / R2 and ``scipy`` Pearson, pooled over every scalar
(row, readout) pair, exactly like ``perturb_lib.evaluators.collection.standard_ones``).

The point is to bound how much of the reported model performance is reachable
without any learned representation.

Usage:
    python scripts/compute_null_baselines.py \
        --shard-root /path/to/plibdata_multiout/<name> \
        --eval-split test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

KEY_COLUMNS = ["dataset_code", "context_code", "perturbation_codes", "log_dose", "time", "split", "n_values"]
VALUE_COLUMNS = KEY_COLUMNS + ["readout_codes", "values"]


@dataclass
class GroupAccumulator:
    """Running per-group sum/count over a flattened ``group_index * n_genes + gene`` key space."""

    name: str
    n_groups: int
    n_genes: int
    sums: np.ndarray = field(init=False)
    counts: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.sums = np.zeros(self.n_groups * self.n_genes, dtype=np.float64)
        self.counts = np.zeros(self.n_groups * self.n_genes, dtype=np.int32)

    @property
    def nbytes(self) -> int:
        return self.sums.nbytes + self.counts.nbytes

    def add(self, group_per_row: np.ndarray, row_index: np.ndarray, genes: np.ndarray, values: np.ndarray) -> None:
        """Accumulate one shard.

        Shards are stored per dataset/cell-line, so only a handful of distinct groups occur in
        any one of them. Binning into a compacted local index keeps the per-shard cost
        proportional to the groups actually present rather than to the full table.
        """
        present, local = np.unique(group_per_row, return_inverse=True)
        local_flat = local[row_index] * self.n_genes + genes
        width = present.size * self.n_genes
        local_sums = np.bincount(local_flat, weights=values, minlength=width)
        local_counts = np.bincount(local_flat, minlength=width)
        for slot, group in enumerate(present):
            lo, hi = slot * self.n_genes, (slot + 1) * self.n_genes
            dst_lo, dst_hi = group * self.n_genes, (group + 1) * self.n_genes
            self.sums[dst_lo:dst_hi] += local_sums[lo:hi]
            self.counts[dst_lo:dst_hi] += local_counts[lo:hi].astype(np.int32)

    def means(self) -> tuple[np.ndarray, np.ndarray]:
        seen = self.counts > 0
        out = np.zeros_like(self.sums)
        np.divide(self.sums, np.maximum(self.counts, 1), out=out)
        return out, seen


class MetricAccumulator:
    """Streaming sufficient statistics for RMSE / MAE / R2 / Pearson.

    R2 and Pearson are computed against the mean of whatever subset is accumulated, so a
    per-dataset accumulator reports variance explained within that dataset.
    """

    def __init__(self) -> None:
        self.n = 0
        self.sse = 0.0
        self.sae = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.sum_p = 0.0
        self.sum_p2 = 0.0
        self.sum_yp = 0.0
        self.n_fallback = 0

    def add(self, y: np.ndarray, p: np.ndarray, n_fallback: int = 0) -> None:
        y = y.astype(np.float64, copy=False)
        d = y - p
        self.n += y.size
        self.sse += float(np.dot(d, d))
        self.sae += float(np.abs(d).sum())
        self.sum_y += float(y.sum())
        self.sum_y2 += float(np.dot(y, y))
        self.sum_p += float(p.sum())
        self.sum_p2 += float(np.dot(p, p))
        self.sum_yp += float(np.dot(y, p))
        self.n_fallback += n_fallback

    def finalize(self) -> dict[str, float]:
        n = self.n
        if n == 0:
            return {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan"),
                    "Pearson": float("nan"), "n_scalars": 0, "frac_backed_off": float("nan")}
        ybar = self.sum_y / n
        ss_tot = self.sum_y2 - n * ybar * ybar
        cov = self.sum_yp - self.sum_y * self.sum_p / n
        var_y = self.sum_y2 - self.sum_y**2 / n
        var_p = self.sum_p2 - self.sum_p**2 / n
        denom = np.sqrt(var_y * var_p)
        return {
            "RMSE": float(np.sqrt(self.sse / n)),
            "MAE": self.sae / n,
            "R2": float(1.0 - self.sse / ss_tot),
            "Pearson": float(cov / denom) if denom > 0 else float("nan"),
            "n_scalars": n,
            "frac_backed_off": self.n_fallback / n,
        }


def iter_shards(shard_root: Path, columns: list[str], split: str | None, max_shards: int | None = None):
    for path in sorted(shard_root.glob("*/shard_*.parquet"))[:max_shards]:
        frame = pl.read_parquet(path, columns=columns)
        if split is not None:
            frame = frame.filter(pl.col("split") == split)
        if len(frame):
            yield path, frame


def flatten(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (gene codes, values, per-scalar row index) for one shard."""
    n_values = frame["n_values"].to_numpy().astype(np.int64)
    genes = frame["readout_codes"].explode().to_numpy().astype(np.int64)
    values = frame["values"].explode().to_numpy().astype(np.float64)
    row_index = np.repeat(np.arange(len(frame), dtype=np.int64), n_values)
    return genes, values, row_index


def dose_bin(log_dose: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin 0 is reserved for zero dose (log_dose == -inf); finite doses go to bins 1..len(edges)."""
    out = np.zeros(log_dose.shape, dtype=np.int64)
    finite = np.isfinite(log_dose)
    out[finite] = 1 + np.searchsorted(edges, log_dose[finite], side="right")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--n-dose-bins", type=int, default=4, help="Finite-dose quantile bins; zero dose gets its own.")
    parser.add_argument("--output", type=Path, default=Path("results/null_baselines.json"))
    parser.add_argument("--max-shards", type=int, default=None, help="Debug: only read the first N shard files.")
    args = parser.parse_args()

    root: Path = args.shard_root
    started = time.time()

    # ---- prepass: key columns only, to size the index spaces and pick dose bins ----
    shard_paths = sorted(root.glob("*/shard_*.parquet"))[: args.max_shards]
    keys = pl.concat([pl.read_parquet(p, columns=KEY_COLUMNS) for p in shard_paths])
    train_keys = keys.filter(pl.col("split") == "train")
    eval_keys = keys.filter(pl.col("split") == args.eval_split)
    if not len(eval_keys):
        raise SystemExit(f"no rows in split {args.eval_split!r}")

    n_datasets = int(keys["dataset_code"].max()) + 1
    n_contexts = int(keys["context_code"].max()) + 1
    times = np.sort(keys["time"].unique().to_numpy())
    time_slot = {float(t): i for i, t in enumerate(times)}

    finite_dose = train_keys["log_dose"].to_numpy()
    finite_dose = finite_dose[np.isfinite(finite_dose)]
    quantiles = np.linspace(0, 1, args.n_dose_bins + 1)[1:-1]
    dose_edges = np.quantile(finite_dose, quantiles)
    n_dose_bins = len(dose_edges) + 2

    # Molecule baselines only need rows for molecules that show up in the eval split.
    eval_molecules = np.sort(eval_keys["perturbation_codes"].explode().unique().to_numpy())
    mol_slot_lookup = {int(m): i for i, m in enumerate(eval_molecules)}
    n_eval_molecules = len(eval_molecules)

    n_genes = 0
    for path in sorted(root.glob("*/metadata.parquet")):
        readouts = pl.read_parquet(path, columns=["readouts"])["readouts"].explode()
        if len(readouts):
            n_genes = max(n_genes, int(readouts.n_unique()))
    # readout codes index a global vocabulary; take the true upper bound from the data itself
    for _path, frame in iter_shards(root, ["readout_codes", "split"], None, args.max_shards):
        n_genes = max(n_genes, int(frame["readout_codes"].explode().max()) + 1)
        break
    n_genes = max(n_genes, 43615)

    print(f"genes={n_genes} datasets={n_datasets} contexts={n_contexts} times={len(times)} dose_bins={n_dose_bins}")
    print(f"eval split={args.eval_split!r} rows={len(eval_keys)} molecules={n_eval_molecules}")
    print(f"dose bin edges (log10): {np.round(dose_edges, 4).tolist()}  (bin 0 = zero dose)")

    accumulators = {
        "gene": GroupAccumulator("gene", 1, n_genes),
        "gene_x_dataset": GroupAccumulator("gene_x_dataset", n_datasets, n_genes),
        "gene_x_context": GroupAccumulator("gene_x_context", n_contexts, n_genes),
        "gene_x_context_x_dose": GroupAccumulator("gene_x_context_x_dose", n_contexts * n_dose_bins, n_genes),
        "gene_x_context_x_time": GroupAccumulator("gene_x_context_x_time", n_contexts * len(times), n_genes),
        "gene_x_molecule": GroupAccumulator("gene_x_molecule", max(n_eval_molecules, 1), n_genes),
    }
    total_bytes = sum(a.nbytes for a in accumulators.values())
    print(f"accumulator memory: {total_bytes / 2**30:.2f} GiB")

    def row_group_indices(frame: pl.DataFrame) -> dict[str, np.ndarray]:
        dataset = frame["dataset_code"].to_numpy().astype(np.int64)
        context = frame["context_code"].to_numpy().astype(np.int64)
        dose = dose_bin(frame["log_dose"].to_numpy().astype(np.float64), dose_edges)
        tslot = np.array([time_slot[float(t)] for t in frame["time"].to_numpy()], dtype=np.int64)
        molecule = frame["perturbation_codes"].list.first().to_numpy().astype(np.int64)
        mol = np.array([mol_slot_lookup.get(int(m), -1) for m in molecule], dtype=np.int64)
        return {
            "gene": np.zeros(len(frame), dtype=np.int64),
            "gene_x_dataset": dataset,
            "gene_x_context": context,
            "gene_x_context_x_dose": context * n_dose_bins + dose,
            "gene_x_context_x_time": context * len(times) + tslot,
            "gene_x_molecule": mol,
        }

    # ---- pass 1: fit group means on train ----
    global_sum = 0.0
    global_count = 0
    n_shards = 0
    for path, frame in iter_shards(root, VALUE_COLUMNS, "train", args.max_shards):
        genes, values, row_index = flatten(frame)
        global_sum += float(values.sum())
        global_count += values.size
        groups = row_group_indices(frame)
        for name, per_row in groups.items():
            if name == "gene_x_molecule":
                keep_rows = per_row >= 0
                if not keep_rows.any():
                    continue
                keep = keep_rows[row_index]
                # reindex the surviving rows so row_index stays contiguous
                remap = np.cumsum(keep_rows) - 1
                accumulators[name].add(per_row[keep_rows], remap[row_index[keep]], genes[keep], values[keep])
            else:
                accumulators[name].add(per_row, row_index, genes, values)
        n_shards += 1
        if n_shards % 20 == 0:
            print(f"  train shard {n_shards}: {global_count / 1e9:.2f}e9 scalars, {time.time() - started:.0f}s", flush=True)
    global_mean = global_sum / global_count
    print(f"train: {global_count} scalars over {n_shards} shards, global mean {global_mean:.6f}")

    fitted = {name: acc.means() for name, acc in accumulators.items()}
    gene_mean, gene_seen = fitted["gene"]

    # ---- pass 2: score on the eval split, pooled and per dataset ----
    baseline_names = list(accumulators) + ["global_mean"]
    metrics = {name: MetricAccumulator() for name in baseline_names}
    per_dataset: dict[int, dict[str, MetricAccumulator]] = {}
    for path, frame in iter_shards(root, VALUE_COLUMNS, args.eval_split, args.max_shards):
        genes, values, row_index = flatten(frame)
        groups = row_group_indices(frame)
        gene_fallback = np.where(gene_seen[genes], gene_mean[genes], global_mean)
        dataset_of_scalar = groups["gene_x_dataset"][row_index]

        predictions = {"global_mean": np.full(values.shape, global_mean)}
        backoffs = {"global_mean": np.zeros(values.shape, dtype=bool)}
        for name, per_row in groups.items():
            means, seen = fitted[name]
            group_of_scalar = per_row[row_index]
            valid = group_of_scalar >= 0
            flat = np.where(valid, group_of_scalar, 0) * n_genes + genes
            hit = seen[flat] & valid
            predictions[name] = np.where(hit, means[flat], gene_fallback)
            backoffs[name] = ~hit

        for name, prediction in predictions.items():
            metrics[name].add(values, prediction, n_fallback=int(backoffs[name].sum()))
        for dataset in np.unique(dataset_of_scalar):
            mask = dataset_of_scalar == dataset
            bucket = per_dataset.setdefault(int(dataset), {n: MetricAccumulator() for n in baseline_names})
            for name, prediction in predictions.items():
                bucket[name].add(values[mask], prediction[mask], n_fallback=int(backoffs[name][mask].sum()))

    results = {name: acc.finalize() for name, acc in metrics.items()}
    ordered = dict(sorted(results.items(), key=lambda kv: kv[1]["RMSE"]))

    print(f"\n=== pooled over all datasets ===")
    print(f"{'baseline':<26} {'RMSE':>8} {'MAE':>8} {'R2':>9} {'Pearson':>8} {'backoff':>8}")
    for name, m in ordered.items():
        print(
            f"{name:<26} {m['RMSE']:8.4f} {m['MAE']:8.4f} {m['R2']:9.4f} "
            f"{m['Pearson']:8.4f} {m['frac_backed_off'] * 100:7.1f}%"
        )

    # Shards are laid out one directory per dataset/cell-line, so each directory pins one code.
    dataset_names: dict[int, str] = {}
    for shard_dir in sorted(p.parent for p in root.glob("*/metadata.parquet")):
        one = pl.read_parquet(sorted(shard_dir.glob("shard_*.parquet"))[0], columns=["dataset_code"])
        names = pl.read_parquet(shard_dir / "metadata.parquet", columns=["datasets"])["datasets"].explode()
        for code in one["dataset_code"].unique().to_list():
            if names.len():
                dataset_names.setdefault(int(code), str(names[0]))

    dataset_results: dict[str, dict[str, Any]] = {}
    print(f"\n=== per dataset ({args.eval_split} split), RMSE / R2 ===")
    print(f"{'dataset':<34} {'scalars':>12} " + " ".join(f"{n.replace('gene_x_', ''):>19}" for n in baseline_names))
    for dataset in sorted(per_dataset, key=lambda d: -per_dataset[d]["gene"].n):
        bucket = {n: acc.finalize() for n, acc in per_dataset[dataset].items()}
        name = dataset_names.get(dataset, f"code_{dataset}")
        dataset_results[name] = {"dataset_code": dataset, **bucket}
        cells = " ".join(f"{bucket[n]['RMSE']:9.4f}/{bucket[n]['R2']:8.4f}" for n in baseline_names)
        print(f"{name[:34]:<34} {bucket['gene']['n_scalars']:12d} {cells}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "shard_root": str(root),
                "eval_split": args.eval_split,
                "n_genes": n_genes,
                "global_mean": global_mean,
                "dose_bin_edges": dose_edges.tolist(),
                "results": ordered,
                "results_by_dataset": dataset_results,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.output}  ({time.time() - started:.0f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Score trained LPM checkpoints per dataset on an on-disk multiout shard root.

Runs the model forward directly from the shard columns (the shard codes are the vocabulary
codes the embedding tables are indexed by) and reports RMSE / MAE / R2 / Pearson pooled and
broken down by dataset, using the same accumulator as ``compute_null_baselines.py`` so the
numbers line up with the groupby baselines.

Usage:
    python scripts/evaluate_checkpoint_by_dataset.py \
        --checkpoint .plib_cache/results/<run>/LPM_*/seed_13/checkpoints/best-*.ckpt \
        --shard-root /path/to/plibdata_multiout/<name> \
        --eval-split test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_null_baselines import VALUE_COLUMNS, MetricAccumulator, iter_shards  # noqa: E402

ZERO_DOSE_SENTINEL = -5.0

ACTIVATIONS = {
    "relu": F.relu,
    "gelu": F.gelu,
    "leaky_relu": lambda x: F.leaky_relu(x, 0.01),
}


class CheckpointModel:
    """Minimal inference-only reconstruction of the LPM forward pass."""

    def __init__(self, checkpoint_path: Path, device: torch.device, offset_file: Path | None = None) -> None:
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = blob["state_dict"]
        hparams = blob.get("hyper_parameters", {}) or {}
        self.activation = ACTIVATIONS[str(hparams.get("activation", "relu"))]
        self.aggregation = str(hparams.get("embedding_aggregation_mode", "mean"))
        self.device = device
        self.w = {k: v.to(device=device, dtype=torch.float32) for k, v in state.items() if torch.is_tensor(v)}
        layer_ids = sorted({int(k.split(".")[1]) for k in self.w if k.startswith("predictor.") and k.endswith(".weight")})
        self.predictor_layers = layer_ids
        self.has_residual_head = "embedding_residual_head.weight" in self.w
        self.n_genes = self.w["predictor.%d.bias" % layer_ids[-1]].shape[0]

        # The groupby offset is a non-persistent buffer, so it is absent from the checkpoint and
        # has to be re-read from the file the run was configured with.
        configured = offset_file or hparams.get("groupby_offset_file")
        self.offset_table = None
        if configured:
            path = Path(configured)
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint was trained with groupby offset {path}, which is missing")
            with np.load(path) as payload:
                table = torch.from_numpy(payload["table"].astype(np.float32))
                self.dose_edges = torch.from_numpy(payload["dose_edges"].astype(np.float32)).to(device)
                self.n_dose_bins = int(payload["n_dose_bins"])
            self.offset_table = table.reshape(-1).to(device)
            print(f"  using groupby offset {path.name}: {table.shape[0] // self.n_dose_bins} cell lines "
                  f"x {self.n_dose_bins} dose bins")

    def neutralize_untrained(self, trained_contexts: set[int], trained_datasets: set[int]) -> None:
        """Replace embedding rows that never saw a gradient with the mean trained row.

        A checkpoint trained on a subset of the corpus leaves the other cell lines' embedding rows
        at their N(0, 1) init, which is far outside the range the predictor was fitted on, so
        scoring it everywhere measures init noise as much as transfer. Substituting the centroid of
        the trained rows keeps those rows in distribution and isolates the molecule pathway.
        """
        for key, trained in (
            ("context_embedding_layer.weight", trained_contexts),
            ("dataset_embedding_layer.weight", trained_datasets),
        ):
            weight = self.w[key]
            mask = torch.ones(weight.shape[0], dtype=torch.bool, device=weight.device)
            mask[torch.tensor(sorted(trained), dtype=torch.long, device=weight.device)] = False
            if not bool(mask.any()):
                continue
            weight[mask] = weight[~mask].mean(dim=0, keepdim=True)
            print(f"  neutralized {int(mask.sum())}/{weight.shape[0]} untrained rows in {key}")

    def offset_for(self, batch: dict[str, torch.Tensor], rows: torch.Tensor, genes: torch.Tensor) -> torch.Tensor:
        log_dose = batch["log_dose_raw"]
        zero_dose = ~torch.isfinite(log_dose) | (log_dose <= ZERO_DOSE_SENTINEL)
        dose = torch.zeros_like(log_dose, dtype=torch.long)
        dosed = ~zero_dose
        dose[dosed] = 1 + torch.searchsorted(self.dose_edges, log_dose[dosed].contiguous(), right=True)
        group = batch["context"] * self.n_dose_bins + dose
        return self.offset_table[group[rows] * self.n_genes + genes]

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        w = self.w
        embedded_dataset = F.embedding(batch["dataset"], w["dataset_embedding_layer.weight"])
        embedded_context = F.embedding(batch["context"], w["context_embedding_layer.weight"])
        embedded_perturb = F.embedding_bag(
            batch["perturbation_flat"],
            w["perturb_embedding_layer.weight"],
            batch["perturbation_offset"],
            mode=self.aggregation,
        )
        dose = batch["log_dose"].unsqueeze(-1)
        time_ = batch["time"].unsqueeze(-1)
        embedded_dose = dose @ w["log_dose_layer.weight"].T + w["log_dose_layer.bias"]
        embedded_time = time_ @ w["time_layer.weight"].T + w["time_layer.bias"]

        hidden = torch.cat(
            [embedded_dataset, embedded_context, embedded_perturb, embedded_dose, embedded_time], dim=1
        )
        for position, layer in enumerate(self.predictor_layers):
            hidden = hidden @ w[f"predictor.{layer}.weight"].T + w[f"predictor.{layer}.bias"]
            if position < len(self.predictor_layers) - 1:
                hidden = self.activation(hidden)
        if self.has_residual_head:
            summed = embedded_dataset + embedded_context + embedded_perturb
            hidden = hidden + summed @ w["embedding_residual_head.weight"].T
        return hidden


def iter_selected_shards(root: Path, split: str, prefixes: tuple[str, ...] | None, max_shards: int | None):
    """``iter_shards`` restricted to shard directories whose name starts with one of ``prefixes``."""
    if not prefixes:
        yield from iter_shards(root, VALUE_COLUMNS, split, max_shards)
        return
    seen = 0
    for path in sorted(root.glob("*/shard_*.parquet")):
        if not path.parent.name.startswith(prefixes):
            continue
        if max_shards is not None and seen >= max_shards:
            return
        seen += 1
        frame = pl.read_parquet(path, columns=VALUE_COLUMNS).filter(pl.col("split") == split)
        if len(frame):
            yield path, frame


def batch_from_frame(frame: pl.DataFrame, lo: int, hi: int, device: torch.device) -> dict[str, torch.Tensor]:
    chunk = frame[lo:hi]
    codes = chunk["perturbation_codes"].to_list()
    lengths = torch.tensor([len(c) for c in codes], dtype=torch.long)
    flat = torch.tensor([code for row in codes for code in row], dtype=torch.long)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]])
    raw_log_dose = torch.tensor(chunk["log_dose"].to_numpy().astype("float32"))
    # Match to_tensor_dict: zero dose arrives as -inf in the parquet and the model is fed a
    # finite sentinel instead. Keep the raw column too, since dose binning depends on it.
    log_dose = torch.nan_to_num(raw_log_dose, neginf=ZERO_DOSE_SENTINEL, posinf=5.0)
    return {
        "dataset": torch.tensor(chunk["dataset_code"].to_numpy().astype("int64")).to(device),
        "context": torch.tensor(chunk["context_code"].to_numpy().astype("int64")).to(device),
        "perturbation_flat": flat.to(device),
        "perturbation_offset": offsets.to(device),
        "log_dose": log_dose.to(device),
        "log_dose_raw": raw_log_dose.to(device),
        "time": torch.tensor(chunk["time"].to_numpy().astype("float32")).to(device),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, nargs="+", type=Path)
    parser.add_argument("--label", nargs="+", default=None, help="Display names, one per checkpoint.")
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument(
        "--groupby-offset",
        type=Path,
        default=None,
        help="Override the offset npz. Defaults to whatever the checkpoint's hparams recorded.",
    )
    parser.add_argument(
        "--eval-source-prefix",
        nargs="+",
        default=None,
        help="Restrict scoring to shard directories with these prefixes, e.g. to score L1000 validation only.",
    )
    parser.add_argument(
        "--trained-source-prefix",
        nargs="+",
        default=None,
        help=(
            "Shard directory prefixes the checkpoint was trained on, e.g. l1000_phase1 l1000_phase2. "
            "Context/dataset embedding rows outside them are replaced by the mean trained row."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("results/checkpoint_by_dataset.json"))
    args = parser.parse_args()

    labels = args.label or [p.parts[2] if len(p.parts) > 2 else p.stem for p in args.checkpoint]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  checkpoints={len(args.checkpoint)}")

    root: Path = args.shard_root
    dataset_names: dict[int, str] = {}
    for shard_dir in sorted(p.parent for p in root.glob("*/metadata.parquet")):
        one = pl.read_parquet(sorted(shard_dir.glob("shard_*.parquet"))[0], columns=["dataset_code"])
        names = pl.read_parquet(shard_dir / "metadata.parquet", columns=["datasets"])["datasets"].explode()
        for code in one["dataset_code"].unique().to_list():
            if names.len():
                dataset_names.setdefault(int(code), str(names[0]))

    eval_prefixes = tuple(args.eval_source_prefix) if args.eval_source_prefix else None
    if eval_prefixes:
        print(f"scoring only shard directories starting with {eval_prefixes}")

    trained_contexts: set[int] = set()
    trained_datasets: set[int] = set()
    if args.trained_source_prefix:
        prefixes = tuple(args.trained_source_prefix)
        for shard_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefixes)):
            frame = pl.concat(
                [pl.read_parquet(p, columns=["context_code", "dataset_code", "split"]) for p in
                 sorted(shard_dir.glob("shard_*.parquet"))]
            ).filter(pl.col("split") == "train")
            trained_contexts.update(int(c) for c in frame["context_code"].unique().to_list())
            trained_datasets.update(int(c) for c in frame["dataset_code"].unique().to_list())
        print(
            f"trained sources {prefixes}: {len(trained_contexts)} cell lines, "
            f"{len(trained_datasets)} datasets received gradient"
        )

    everything: dict[str, dict] = {}
    for checkpoint_path, label in zip(args.checkpoint, labels):
        started = time.time()
        print(f"\nloading {label}")
        model = CheckpointModel(checkpoint_path, device, args.groupby_offset)
        if trained_contexts:
            model.neutralize_untrained(trained_contexts, trained_datasets)
        pooled = MetricAccumulator()
        per_dataset: dict[int, MetricAccumulator] = {}
        with torch.no_grad():
            for _path, frame in iter_selected_shards(root, args.eval_split, eval_prefixes, args.max_shards):
                dataset_code = int(frame["dataset_code"][0])
                bucket = per_dataset.setdefault(dataset_code, MetricAccumulator())
                for lo in range(0, len(frame), args.row_chunk):
                    hi = min(lo + args.row_chunk, len(frame))
                    chunk = frame[lo:hi]
                    batch = batch_from_frame(frame, lo, hi, device)
                    prediction = model.forward(batch)
                    counts = torch.tensor(chunk["n_values"].to_numpy().astype("int64"))
                    genes = torch.tensor(
                        np.concatenate(chunk["readout_codes"].to_list()).astype("int64")
                    ).to(device)
                    rows = torch.repeat_interleave(
                        torch.arange(hi - lo, dtype=torch.long), counts
                    ).to(device)
                    selected_t = prediction[rows, genes]
                    if model.offset_table is not None:
                        selected_t = selected_t + model.offset_for(batch, rows, genes)
                    selected = selected_t.cpu().numpy().astype(np.float64)
                    truth = np.concatenate(chunk["values"].to_list()).astype(np.float64)
                    pooled.add(truth, selected)
                    bucket.add(truth, selected)

        result = {"pooled": pooled.finalize(), "by_dataset": {}}
        for code, acc in per_dataset.items():
            result["by_dataset"][dataset_names.get(code, f"code_{code}")] = acc.finalize()
        everything[label] = result
        m = result["pooled"]
        print(
            f"\n=== {label} ===  ({time.time() - started:.0f}s)\n"
            f"  pooled: RMSE {m['RMSE']:.4f}  MAE {m['MAE']:.4f}  R2 {m['R2']:.4f}  Pearson {m['Pearson']:.4f}"
        )
        for name, dm in sorted(result["by_dataset"].items(), key=lambda kv: -kv[1]["n_scalars"]):
            print(
                f"    {name[:34]:<34} n={dm['n_scalars']:>11}  RMSE {dm['RMSE']:.4f}  R2 {dm['R2']:8.4f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(everything, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

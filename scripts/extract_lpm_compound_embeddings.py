#!/usr/bin/env python3
"""Extract learned compound embeddings from an LPM checkpoint.

The LPM stores compound embeddings in ``perturb_embedding_layer.weight`` and
the corresponding row labels in ``perturb_symbols``. This script reads either
a PyTorch-Lightning ``.ckpt`` or the repo's ``model.pt`` tuple format and
writes:

  * ``compound_embeddings.npy``: float32 matrix, rows aligned to metadata
  * ``compound_metadata.parquet`` and ``compound_metadata.tsv``: code/symbol rows
  * ``df_pert.pkl``: notebook-compatible pandas frame with list-valued vectors
  * ``manifest.json``: source path, epoch/step when available, shape
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


DEFAULT_CHECKPOINT = Path(
    ".plib_cache/results/lpm_multiout_all_data_170epoch_lustre/"
    "LPM_7952bf9c6dc64c60/seed_13/checkpoints/epoch-epoch=0169.ckpt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to a Lightning .ckpt or repo model.pt file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write extracted embeddings. Defaults next to the checkpoint/run.",
    )
    parser.add_argument(
        "--skip-pickle",
        action="store_true",
        help="Skip writing notebook-compatible df_pert.pkl.",
    )
    return parser.parse_args()


def load_state(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    metadata: dict[str, Any] = {"source": str(path)}

    if isinstance(raw, dict) and "state_dict" in raw:
        metadata["format"] = "lightning_checkpoint"
        metadata["epoch"] = raw.get("epoch")
        metadata["global_step"] = raw.get("global_step")
        return raw["state_dict"], metadata

    if isinstance(raw, tuple) and len(raw) == 3:
        model_id, model_args, state = raw
        metadata["format"] = "model_pt_tuple"
        metadata["model_id"] = model_id
        metadata["model_args"] = model_args
        return state, metadata

    raise ValueError(f"Unsupported checkpoint format: {path}")


def infer_output_dir(checkpoint: Path) -> Path:
    if checkpoint.name == "model.pt":
        return checkpoint.parent / "compound_embeddings"
    if checkpoint.parent.name == "checkpoints":
        stem = checkpoint.stem.replace("-", "_").replace("=", "_")
        return checkpoint.parent.parent / f"compound_embeddings_{stem}"
    return checkpoint.with_suffix("").parent / f"{checkpoint.stem}_compound_embeddings"


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    state, metadata = load_state(checkpoint)
    symbols = list(state["perturb_symbols"])
    embeddings = state["perturb_embedding_layer.weight"].detach().cpu().numpy().astype(np.float32)
    if len(symbols) != embeddings.shape[0]:
        raise ValueError(
            f"Symbol count ({len(symbols)}) does not match embedding rows ({embeddings.shape[0]})."
        )

    output_dir = args.output_dir or infer_output_dir(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.DataFrame({"code": np.arange(len(symbols), dtype=np.int32), "symbol": symbols})
    np.save(output_dir / "compound_embeddings.npy", embeddings)
    metadata_df.to_parquet(output_dir / "compound_metadata.parquet", index=False)
    metadata_df.to_csv(output_dir / "compound_metadata.tsv", sep="\t", index=False)

    if not args.skip_pickle:
        df_pert = metadata_df.copy()
        df_pert["lpm_style_embeddings"] = embeddings.tolist()
        df_pert.to_pickle(output_dir / "df_pert.pkl")

    manifest = {
        **metadata,
        "n_compounds": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "outputs": {
            "embeddings_npy": str(output_dir / "compound_embeddings.npy"),
            "metadata_parquet": str(output_dir / "compound_metadata.parquet"),
            "metadata_tsv": str(output_dir / "compound_metadata.tsv"),
            "df_pert_pickle": None if args.skip_pickle else str(output_dir / "df_pert.pkl"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"Extracted {embeddings.shape[0]} compound embeddings "
        f"with dim={embeddings.shape[1]} to {output_dir}"
    )


if __name__ == "__main__":
    main()

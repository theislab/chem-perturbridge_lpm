#!/usr/bin/env python3
"""Build frozen Morgan fingerprint perturbation embeddings for LPM vocabularies.

The current LPM training data uses PubChem CID strings as perturbation symbols.
This script resolves those CIDs to SMILES, computes Morgan fingerprints with
RDKit, and writes the same embedding artifact layout consumed by LPM's
`pretrained_perturbation_embeddings_path`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOCAB = Path(
    "/lustre/groups/ml01/workspace/olga.novitskaia/lpm_style/.plib_cache/plibdata_multiout/"
    "lpm_multiout_all_data_plus_tahoe_novartis_op3_molholdout_h100_2x_bs4096_200epoch_lustre/"
    "vocab/perturb_vocab.parquet"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".plib_cache" / "morgan_perturbation_embeddings" / "pubchem_morgan_radius2_nbits128"

LOCAL_SMILES_FILES = [
    Path("/lustre/groups/ml01/workspace/olga.novitskaia/login_node_files/op3_signatures/additional_files/df_compounds.csv"),
    Path("/lustre/groups/ml01/workspace/olga.novitskaia/login_node_files/op3_signatures/additional_files/dili_pubchem.csv"),
    Path("/lustre/groups/ml01/workspace/olga.novitskaia/login_node_files/op3_signatures/files/df_pubchem_op3.csv"),
    Path("/lustre/groups/ml01/workspace/olga.novitskaia/login_node_files/op3_signatures/data_tmp/sciplex/df_compounds.csv"),
    Path("/lustre/groups/ml01/workspace/olga.novitskaia/login_node_files/op3_signatures/data_tmp/novartis/df_compounds.csv"),
    Path("/lustre/groups/ml01/workspace/olga.novitskaia/login_node_files/op3_signatures/data_tmp/tahoe/df_compounds.csv"),
    Path("/lustre/groups/ml01/workspace/olga.novitskaia/login_node_files/op3_signatures/data_tmp/vcpi_0001/df_compounds.csv"),
    Path("/lustre/groups/ml01/workspace/olga.novitskaia/login_node_files/op3_signatures/data_tmp/vcpi_0002/df_compounds.csv"),
    Path("/lustre/groups/ml01/workspace/olga.novitskaia/data_updated/vcpi_0001/raw/df_compounds.csv"),
    Path("/lustre/groups/ml01/workspace/olga.novitskaia/data_updated/vcpi_0002/raw/df_compounds.csv"),
    Path("/lustre/groups/ml01/workspace/olga.novitskaia/data_updated/gdpx2/raw/gdpx2_compounds.csv"),
]
CID_COLUMNS = ("pubchem_cid", "cid", "CID", "PubChem CID", "user_compound_id")
SMILES_COLUMNS = ("canonical_smiles", "smiles", "SMILES", "CanonicalSMILES", "ConnectivitySMILES")


def normalize_cid(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return str(int(float(text)))
    except ValueError:
        return None


def read_vocab(path: Path) -> pd.DataFrame:
    vocab = pd.read_parquet(path).sort_values("code").reset_index(drop=True)
    required = {"symbol", "code"}
    missing = required.difference(vocab.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return vocab


def add_smiles(mapping: dict[str, str], sources: dict[str, str], cid: str | None, smiles: Any, source: str) -> None:
    if cid is None:
        return
    if smiles is None:
        return
    smiles_text = str(smiles).strip()
    if not smiles_text or smiles_text.lower() in {"nan", "none", "null", "restricted"}:
        return
    if cid not in mapping:
        mapping[cid] = smiles_text
        sources[cid] = source


def load_local_smiles(files: list[Path]) -> tuple[dict[str, str], dict[str, str], list[str]]:
    mapping: dict[str, str] = {}
    sources: dict[str, str] = {}
    used: list[str] = []
    for path in files:
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False)
        except Exception:
            continue
        cid_col = next((col for col in CID_COLUMNS if col in df.columns), None)
        smiles_col = next((col for col in SMILES_COLUMNS if col in df.columns), None)
        if cid_col is None or smiles_col is None:
            continue
        used.append(str(path))
        for cid_value, smiles in zip(df[cid_col], df[smiles_col]):
            add_smiles(mapping, sources, normalize_cid(cid_value), smiles, f"local:{path}")
    return mapping, sources, used


def load_cache(path: Path, mapping: dict[str, str], sources: dict[str, str]) -> None:
    if not path.is_file():
        return
    df = pd.read_csv(path, sep="\t", dtype=str)
    if not {"pubchem_cid", "smiles"}.issubset(df.columns):
        return
    for _, row in df.iterrows():
        add_smiles(mapping, sources, normalize_cid(row["pubchem_cid"]), row["smiles"], row.get("source", "cache"))


def save_cache(path: Path, mapping: dict[str, str], sources: dict[str, str]) -> None:
    rows = [
        {"pubchem_cid": cid, "smiles": smiles, "source": sources.get(cid, "unknown")}
        for cid, smiles in sorted(mapping.items(), key=lambda item: int(item[0]))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def parse_pubchem_properties(response_json: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in response_json.get("PropertyTable", {}).get("Properties", []):
        cid = normalize_cid(row.get("CID"))
        smiles = row.get("SMILES") or row.get("CanonicalSMILES") or row.get("ConnectivitySMILES")
        if cid is not None and smiles:
            out[cid] = str(smiles)
    return out


def fetch_pubchem_batch(cids: list[str], timeout: int, retries: int) -> tuple[dict[str, str], set[str]]:
    if not cids:
        return {}, set()
    cid_text = ",".join(cids)
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{cid_text}/property/CanonicalSMILES,IsomericSMILES/JSON"
    )
    last_error = ""
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return parse_pubchem_properties(response.json()), set()
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code == 404:
                break
        except Exception as exc:  # noqa: BLE001 - keep fetch robust for batch retries.
            last_error = repr(exc)
        time.sleep(min(5.0, 0.5 * (attempt + 1)))
    if len(cids) == 1:
        return {}, set(cids)
    midpoint = len(cids) // 2
    left, left_missing = fetch_pubchem_batch(cids[:midpoint], timeout=timeout, retries=retries)
    right, right_missing = fetch_pubchem_batch(cids[midpoint:], timeout=timeout, retries=retries)
    merged = {**left, **right}
    missing = left_missing | right_missing
    if not merged and not missing:
        print(f"Warning: PubChem batch failed without resolved/missing CIDs ({last_error})")
    return merged, missing


def fetch_pubchem_smiles(
    cids: list[str],
    mapping: dict[str, str],
    sources: dict[str, str],
    cache_path: Path,
    batch_size: int,
    sleep_seconds: float,
    timeout: int,
    retries: int,
) -> set[str]:
    missing_after_fetch: set[str] = set()
    batches = [cids[i : i + batch_size] for i in range(0, len(cids), batch_size)]
    for idx, batch in enumerate(batches, start=1):
        unresolved = [cid for cid in batch if cid not in mapping]
        if not unresolved:
            continue
        fetched, missing = fetch_pubchem_batch(unresolved, timeout=timeout, retries=retries)
        for cid, smiles in fetched.items():
            add_smiles(mapping, sources, cid, smiles, "pubchem")
        missing_after_fetch.update(missing)
        if idx % 20 == 0 or idx == len(batches):
            print(f"PubChem progress: {idx}/{len(batches)} batches; mapped={len(mapping)}; missing={len(missing_after_fetch)}")
            save_cache(cache_path, mapping, sources)
        time.sleep(sleep_seconds)
    save_cache(cache_path, mapping, sources)
    return missing_after_fetch


def morgan_vector(smiles: str, nbits: int, radius: int) -> np.ndarray | None:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fingerprint = generator.GetFingerprint(mol)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nbits", type=int, default=128)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--missing-policy", choices=["zero", "error"], default="zero")
    args = parser.parse_args()

    vocab = read_vocab(args.vocab)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "cid_smiles_cache.tsv"

    smiles_by_cid, source_by_cid, local_sources = load_local_smiles(LOCAL_SMILES_FILES)
    load_cache(cache_path, smiles_by_cid, source_by_cid)

    numeric_cids: list[str] = []
    for symbol in vocab["symbol"].astype(str):
        cid = normalize_cid(symbol)
        if cid is not None:
            numeric_cids.append(cid)
    numeric_cids = sorted(set(numeric_cids), key=int)
    missing_before_fetch = [cid for cid in numeric_cids if cid not in smiles_by_cid]
    print(
        f"Loaded {len(smiles_by_cid)} CID->SMILES mappings before PubChem fetch; "
        f"need {len(numeric_cids)} numeric CIDs; missing={len(missing_before_fetch)}."
    )
    pubchem_missing = fetch_pubchem_smiles(
        missing_before_fetch,
        smiles_by_cid,
        source_by_cid,
        cache_path,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        timeout=args.timeout,
        retries=args.retries,
    )

    embeddings = np.zeros((len(vocab), args.nbits), dtype=np.float32)
    metadata_rows: list[dict[str, Any]] = []
    missing_smiles_rows: list[dict[str, Any]] = []
    invalid_smiles_rows: list[dict[str, Any]] = []

    for row_idx, row in vocab.iterrows():
        symbol = str(row["symbol"])
        cid = normalize_cid(symbol)
        smiles = smiles_by_cid.get(cid) if cid is not None else None
        source = source_by_cid.get(cid, "") if cid is not None else "control"
        status = "ok"
        on_bits = 0
        if cid is None:
            status = "control_zero"
        elif smiles is None:
            status = "missing_smiles_zero"
            missing_smiles_rows.append({"symbol": symbol, "pubchem_cid": cid})
            if args.missing_policy == "error":
                raise ValueError(f"Missing SMILES for CID {cid}")
        else:
            vector = morgan_vector(smiles, nbits=args.nbits, radius=args.radius)
            if vector is None:
                status = "invalid_smiles_zero"
                invalid_smiles_rows.append({"symbol": symbol, "pubchem_cid": cid, "smiles": smiles})
                if args.missing_policy == "error":
                    raise ValueError(f"RDKit failed to parse SMILES for CID {cid}: {smiles}")
            else:
                embeddings[int(row_idx)] = vector
                on_bits = int(vector.sum())
        metadata_rows.append(
            {
                "code": int(row["code"]),
                "symbol": symbol,
                "pubchem_cid": "" if cid is None else cid,
                "smiles": "" if smiles is None else smiles,
                "smiles_source": source,
                "fingerprint_status": status,
                "fingerprint_on_bits": on_bits,
            }
        )

    metadata = pd.DataFrame(metadata_rows)
    metadata_parquet = output_dir / "compound_metadata.parquet"
    metadata_tsv = output_dir / "compound_metadata.tsv"
    embeddings_npy = output_dir / "compound_embeddings.npy"
    pickle_path = output_dir / "df_pert.pkl"
    manifest_path = output_dir / "manifest.json"

    metadata.to_parquet(metadata_parquet, index=False)
    metadata.to_csv(metadata_tsv, sep="\t", index=False)
    np.save(embeddings_npy, embeddings)
    df_pert = metadata[["code", "symbol"]].copy()
    df_pert["lpm_style_embeddings"] = embeddings.tolist()
    df_pert.to_pickle(pickle_path)

    if missing_smiles_rows:
        pd.DataFrame(missing_smiles_rows).to_csv(output_dir / "missing_smiles.tsv", sep="\t", index=False)
    if invalid_smiles_rows:
        pd.DataFrame(invalid_smiles_rows).to_csv(output_dir / "invalid_smiles.tsv", sep="\t", index=False)

    counts = metadata["fingerprint_status"].value_counts().to_dict()
    manifest = {
        "embedding_type": "pubchem_morgan_fingerprint",
        "vocab": str(args.vocab),
        "radius": args.radius,
        "nbits": args.nbits,
        "n_compounds": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "status_counts": {str(key): int(value) for key, value in counts.items()},
        "pubchem_missing_after_fetch": sorted(pubchem_missing, key=int),
        "local_smiles_sources": local_sources,
        "outputs": {
            "df_pert_pickle": str(pickle_path),
            "metadata_parquet": str(metadata_parquet),
            "metadata_tsv": str(metadata_tsv),
            "embeddings_npy": str(embeddings_npy),
            "manifest_json": str(manifest_path),
            "cid_smiles_cache": str(cache_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

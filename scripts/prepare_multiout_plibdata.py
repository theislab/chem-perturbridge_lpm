#!/usr/bin/env python3
"""Prepare a config-specific multi-output plibdata root before training.

The launcher uses this as an idempotent preflight: it checks the exact source
list from the training config, refuses to reuse a shard root created for a
different subset, and builds only missing source shards under a root-level lock.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_multiout_plibdata.py"
CONFIG_DIR = REPO_ROOT / "perturb_gym" / "configs" / "collection"
DEFAULT_FALLBACK_CACHE_ROOTS = (
    Path("/lustre/groups/ml01/workspace/olga.novitskaia/login_node_files/lpm_style/.plib_cache"),
)
VOCAB_FILES = (
    "dataset_vocab.parquet",
    "context_vocab.parquet",
    "perturb_vocab.parquet",
    "readout_vocab.parquet",
)


def log(message: str) -> None:
    print(f"[prepare_multiout_plibdata] {message}", flush=True)


def resolve_config_path(value: str) -> Path:
    candidate = Path(value)
    candidates = [candidate, REPO_ROOT / candidate]
    if candidate.suffix != ".yaml":
        candidates.append(CONFIG_DIR / f"{value}.yaml")
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Could not find config '{value}'")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


def multiout_enabled(config: dict[str, Any]) -> bool:
    for model_config in config.get("model_configs", []):
        model_args = model_config.get("model_args", {})
        if model_args.get("output_mode") == "multiout":
            return True
    return False


def data_config(config: dict[str, Any]) -> dict[str, Any]:
    data_configs = config.get("data_configs", [])
    if len(data_configs) != 1:
        raise ValueError(f"Expected exactly one data config, found {len(data_configs)}")
    return data_configs[0]


def infer_cache_root(output_root: Path) -> Path:
    if output_root.parent.name == "plibdata_multiout":
        return output_root.parent.parent
    if output_root.name == "plibdata":
        return output_root.parent
    return output_root.parent


def split_env_roots(value: str | None) -> list[Path]:
    if not value:
        return []
    roots: list[Path] = []
    for chunk in value.replace(",", os.pathsep).split(os.pathsep):
        chunk = chunk.strip()
        if chunk:
            roots.append(Path(chunk))
    return roots


def fallback_cache_roots(args: argparse.Namespace) -> list[Path]:
    roots: list[Path] = []
    roots.extend(args.fallback_cache_root or [])
    roots.extend(split_env_roots(os.environ.get("DATA_PREP_FALLBACK_CACHE_ROOTS")))
    roots.extend(split_env_roots(os.environ.get("FALLBACK_CACHE_ROOT")))
    roots.extend(path for path in DEFAULT_FALLBACK_CACHE_ROOTS if path.is_dir())

    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        resolved = resolve_path(root)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def parse_sources_tsv(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    sources: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines()):
        if line_number == 0:
            if line != "index\tsource":
                raise RuntimeError(f"Unexpected header in {path}: {line!r}")
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise RuntimeError(f"Malformed source row in {path}: {line!r}")
        sources.append(fields[1])
    return sources


def validate_output_root(output_root: Path, sources: list[str]) -> None:
    existing_sources = parse_sources_tsv(output_root / "sources.tsv")
    if existing_sources is not None and existing_sources != sources:
        raise RuntimeError(
            f"{output_root} already contains sources.tsv for a different dataset subset. "
            "Give each subset a distinct on_disk_shard_root; refusing to overwrite."
        )

    if not output_root.exists():
        return
    source_set = set(sources)
    unexpected_dirs = []
    for path in output_root.iterdir():
        if not path.is_dir() or path.name.startswith(".") or path.name == "vocab":
            continue
        if path.name not in source_set:
            unexpected_dirs.append(path.name)
    if unexpected_dirs:
        preview = ", ".join(sorted(unexpected_dirs)[:5])
        raise RuntimeError(
            f"{output_root} contains source directories that are not in this config: {preview}. "
            "Refusing to mix dataset subsets in one shard root."
        )


def missing_source_indices(output_root: Path, sources: list[str]) -> list[int]:
    needs_scaffolding = not (output_root / "sources.tsv").is_file()
    needs_vocab = not all((output_root / "vocab" / name).is_file() for name in VOCAB_FILES)
    missing = [
        idx
        for idx, source in enumerate(sources)
        if not (output_root / source / "source_manifest.yaml").is_file()
        or not (output_root / source / "metadata.parquet").is_file()
    ]
    if needs_scaffolding or needs_vocab:
        return list(range(len(sources)))
    return missing


def int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def default_jobs() -> int:
    cpus = int_env("SLURM_CPUS_PER_TASK", 1)
    return max(1, min(4, cpus // 8 if cpus >= 8 else 1))


def build_base_command(
    args: argparse.Namespace,
    config_path: Path,
    output_root: Path,
    cache_root: Path,
    reference_plibdata_root: Path | None,
    fallbacks: list[Path],
) -> list[str]:
    command = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
        "--cache-root",
        str(cache_root),
        "--target-values-per-shard",
        str(args.target_values_per_shard),
        "--value-dtype",
        args.value_dtype,
        "--float-feature-dtype",
        args.float_feature_dtype,
        "--skip-existing",
    ]
    if reference_plibdata_root is not None:
        command.extend(["--reference-plibdata-root", str(reference_plibdata_root)])
    for fallback in fallbacks:
        command.extend(["--fallback-cache-root", str(fallback)])
    return command


def child_env(jobs: int, threads_per_job: int | None) -> dict[str, str]:
    env = os.environ.copy()
    cpus = int_env("SLURM_CPUS_PER_TASK", 1)
    threads = threads_per_job or max(1, min(8, cpus // max(1, jobs)))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
        env[key] = str(threads)
    return env


def run_builds(indices: list[int], base_command: list[str], jobs: int, env: dict[str, str]) -> None:
    if not indices:
        return
    pending = list(indices)
    running: list[tuple[int, subprocess.Popen]] = []
    completed = 0
    total = len(indices)
    log(f"building {total} missing source shard(s) with jobs={jobs}")

    while pending or running:
        while pending and len(running) < jobs:
            source_index = pending.pop(0)
            command = base_command + ["--source-index", str(source_index)]
            log(f"starting source-index {source_index}")
            running.append((source_index, subprocess.Popen(command, cwd=REPO_ROOT, env=env)))

        time.sleep(2)
        still_running: list[tuple[int, subprocess.Popen]] = []
        for source_index, process in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((source_index, process))
                continue
            if return_code != 0:
                for _, other_process in running:
                    if other_process.poll() is None:
                        other_process.terminate()
                raise subprocess.CalledProcessError(return_code, base_command + ["--source-index", str(source_index)])
            completed += 1
            log(f"completed source-index {source_index} ({completed}/{total})")
        running = still_running


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Config id or YAML path.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--reference-plibdata-root", type=Path, default=None)
    parser.add_argument("--fallback-cache-root", action="append", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=int_env("DATA_PREP_JOBS", default_jobs()))
    parser.add_argument("--threads-per-job", type=int, default=None)
    parser.add_argument("--target-values-per-shard", type=int, default=int_env("DATA_PREP_TARGET_VALUES_PER_SHARD", 8_000_000))
    parser.add_argument("--value-dtype", choices=["float32", "float64"], default=os.environ.get("DATA_PREP_VALUE_DTYPE", "float32"))
    parser.add_argument(
        "--float-feature-dtype",
        choices=["float32", "float64"],
        default=os.environ.get("DATA_PREP_FLOAT_FEATURE_DTYPE", "float32"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    if not multiout_enabled(config):
        log(f"{config_path} is not a multi-output config; skipping preprocessing")
        return 0

    dc = data_config(config)
    sources = list(dc["on_disk_data_sources"])
    output_root = resolve_path(args.output_root or Path(dc["on_disk_shard_root"]))
    cache_root_env = os.environ.get("DATA_PREP_CACHE_ROOT")
    cache_root_arg = args.cache_root or (Path(cache_root_env) if cache_root_env else None)
    cache_root = resolve_path(cache_root_arg) if cache_root_arg is not None else infer_cache_root(output_root)
    reference_plibdata_root = args.reference_plibdata_root
    if reference_plibdata_root is None:
        inferred_reference = cache_root / "plibdata"
        reference_plibdata_root = inferred_reference if inferred_reference.is_dir() else None
    else:
        reference_plibdata_root = resolve_path(reference_plibdata_root)
    fallbacks = fallback_cache_roots(args)

    log(f"config={config_path}")
    log(f"output_root={output_root}")
    log(f"cache_root={cache_root}")
    if reference_plibdata_root is not None:
        log(f"reference_plibdata_root={reference_plibdata_root}")
    for fallback in fallbacks:
        log(f"fallback_cache_root={fallback}")
    log(f"sources={len(sources)}")

    base_command = build_base_command(args, config_path, output_root, cache_root, reference_plibdata_root, fallbacks)
    if args.dry_run:
        subprocess.check_call(base_command + ["--dry-run"], cwd=REPO_ROOT)
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".prepare.lock"
    log(f"waiting for data-prep lock {lock_path}")
    with open(lock_path, "w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        log("acquired data-prep lock")
        validate_output_root(output_root, sources)
        missing = missing_source_indices(output_root, sources)
        if not missing:
            log("all configured source shards are already present")
            return 0
        jobs = max(1, args.jobs)
        run_builds(missing, base_command, jobs, child_env(jobs, args.threads_per_job))
        validate_output_root(output_root, sources)
        remaining = missing_source_indices(output_root, sources)
        if remaining:
            raise RuntimeError(f"Data preparation finished with {len(remaining)} source(s) still missing")
        log("data preparation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

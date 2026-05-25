#!/usr/bin/env python3
"""Submit Morgan paper10 source/scratch jobs and dependent fine-tune selectors."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

VARIANTS: dict[str, dict[str, Any]] = {
    "morgan_fixed": {
        "source_family": "all_datasets_morgan_fixed",
        "scratch_family": "scratch_target_only_morgan_fixed",
        "job_prefix": "p10_mfix",
    },
    "morgan_learned": {
        "source_family": "all_datasets_morgan_learned",
        "scratch_family": "",
        "job_prefix": "p10_mlrn",
    },
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def submit(cmd: list[str], dry_run: bool = False) -> str:
    print(" ".join(cmd))
    if dry_run:
        return "DRYRUN"
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    job_id = completed.stdout.strip().split(";")[0]
    print(job_id)
    return job_id


def submit_training(row: dict[str, str], variant: dict[str, Any], dry_run: bool = False) -> str:
    family = row["model_family"]
    seed = row["seed"]
    slug = row["dataset_slug"]
    if family == variant["source_family"]:
        job_name = f"{variant['job_prefix']}_all_s{seed}"
    elif family == variant["scratch_family"]:
        job_name = f"{variant['job_prefix']}_scr_{slug}_s{seed}"
    else:
        raise ValueError(f"Unexpected model family for variant: {family}")
    return submit(
        [
            "sbatch",
            "--parsable",
            "--time=24:00:00",
            f"--gpus={row['gpus']}",
            f"--job-name={job_name[:120]}",
            "--exclude=supergpu17,supergpu32",
            "--export=ALL,PREPARE_MULTI_OUTPUT_DATA=0,CLEAN_RESULTS=1,PERTURB_GYM_EXTRACT_COMPOUND_EMBEDDINGS=0",
            "run_single_node_lightning.sh",
            "--config",
            row["config_id"],
        ],
        dry_run=dry_run,
    )


def submit_selector(variant_name: str, source_job_ids: list[str], manifest: Path, jobs_manifest: Path, dry_run: bool) -> str:
    dependency = "afterok:" + ":".join(job_id for job_id in source_job_ids if job_id != "DRYRUN")
    cmd = ["sbatch", "--parsable", f"--job-name=p10_{variant_name}_select"]
    if dependency != "afterok:":
        cmd.append(f"--dependency={dependency}")
    cmd.extend(
        [
            "run_lpm_paper10_morgan_select_and_submit.sh",
            "--variant",
            variant_name,
            "--config-manifest",
            str(manifest),
            "--source-scratch-jobs",
            str(jobs_manifest),
        ]
    )
    return submit(cmd, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    manifest = args.manifest or RESULTS_DIR / f"lpm_paper10_{args.variant}_config_manifest.tsv"
    output = args.output or RESULTS_DIR / f"lpm_paper10_{args.variant}_source_scratch_slurm_jobs.tsv"
    rows = read_tsv(manifest)

    submit_rows: list[dict[str, str]] = []
    allowed_families = {variant["source_family"], variant["scratch_family"]}
    allowed_families.discard("")
    for row in rows:
        if row["model_family"] not in allowed_families:
            continue
        job_id = submit_training(row, variant, dry_run=args.dry_run)
        submit_rows.append({**row, "job_id": job_id})

    write_tsv(output, submit_rows)
    source_job_ids = [row["job_id"] for row in submit_rows if row["model_family"] == variant["source_family"]]
    selector_job_id = submit_selector(args.variant, source_job_ids, manifest, output, dry_run=args.dry_run)
    selector_manifest = RESULTS_DIR / f"lpm_paper10_{args.variant}_selector_slurm_job.tsv"
    write_tsv(selector_manifest, [{"job_id": selector_job_id}])
    print(f"Wrote {output}")
    print(f"Wrote {selector_manifest}")


if __name__ == "__main__":
    main()

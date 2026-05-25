#!/usr/bin/env python3
"""Submit paper10 all-dataset source and scratch target-only jobs."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_MANIFEST = RESULTS_DIR / "lpm_paper10_config_manifest.tsv"


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


def submit_training(row: dict[str, str], dry_run: bool = False) -> str:
    family = row["model_family"]
    seed = row["seed"]
    slug = row["dataset_slug"]
    if family == "all_datasets":
        job_name = f"p10_all_s{seed}"
    else:
        job_name = f"p10_scr_{slug}_s{seed}"
    return submit(
        [
            "sbatch",
            "--parsable",
            "--time=24:00:00",
            f"--gpus={row['gpus']}",
            f"--job-name={job_name[:120]}",
            "--exclude=supergpu17",
            "--export=ALL,PREPARE_MULTI_OUTPUT_DATA=0,CLEAN_RESULTS=1,PERTURB_GYM_EXTRACT_COMPOUND_EMBEDDINGS=0",
            "run_single_node_lightning.sh",
            "--config",
            row["config_id"],
        ],
        dry_run=dry_run,
    )


def submit_selector(source_job_ids: list[str], jobs_manifest: Path, dry_run: bool = False) -> str:
    dependency = "afterok:" + ":".join(job_id for job_id in source_job_ids if job_id != "DRYRUN")
    cmd = ["sbatch", "--parsable"]
    if dependency != "afterok:":
        cmd.append(f"--dependency={dependency}")
    cmd.extend(["run_lpm_paper10_select_and_submit.sh", "--scratch-jobs", str(jobs_manifest)])
    return submit(cmd, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "lpm_paper10_source_scratch_slurm_jobs.tsv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = read_tsv(args.manifest)
    submit_rows: list[dict[str, str]] = []
    for row in rows:
        if row["model_family"] not in {"all_datasets", "scratch_target_only"}:
            continue
        job_id = submit_training(row, dry_run=args.dry_run)
        submit_rows.append({**row, "job_id": job_id})

    write_tsv(args.output, submit_rows)
    source_job_ids = [row["job_id"] for row in submit_rows if row["model_family"] == "all_datasets"]
    selector_job_id = submit_selector(source_job_ids, args.output, dry_run=args.dry_run)
    selector_manifest = RESULTS_DIR / "lpm_paper10_selector_slurm_job.tsv"
    write_tsv(selector_manifest, [{"job_id": selector_job_id}])
    print(f"Wrote {args.output}")
    print(f"Wrote {selector_manifest}")


if __name__ == "__main__":
    main()

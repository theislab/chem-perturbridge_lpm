# LPM-Style — Quick Reference

This is a modified fork of [perturblib/perturblib](https://github.com/perturblib/perturblib),
wired for multi-node DDP training of the 6-feature LPM on Slurm. The original library
README is preserved here as [README_original.md](README_original.md).

## Quick start

```bash
sbatch run.sh                  # train with the YAML named in run.sh (default: lpm_modified)
tail -f perturb_lib.<jobid>.out
```

Use the checklist below once per machine or workflow change; the sections that follow expand each block in order (`Environment` → `Submitting` → `Data shards` → `Config`).

## Checklist before you submit

- [ ] **Mamba env path:** `run.sh` ~line 120 (`mamba activate ~/deg_venv`) matches where you created the env (`mamba env create --prefix ~/deg_venv --file env.yml`). Change both if you use another path.
- [ ] **`run.sh` Slurm / resources:** In the `#SBATCH` block, check `#SBATCH --nodes`, `--ntasks-per-node`, `--gpus-per-task`, `--cpus-per-task`, `--mem`, `-t` (walltime), `--qos`, `--partition`, and `--exclude` for your cluster and queue.
- [ ] **`run.sh` config id:** Bottom of `run.sh` passes `--config_file_id_or_path=...` (e.g. `lpm_modified` → `perturb_gym/configs/collection/lpm_modified.yaml`). Set this to the config you intend to train.
- [ ] **Data on disk:** Shard directories exist under your project (commonly `.plib_cache/plibdata/<dataset_folder>/…` with `metadata.parquet` and `shard_*.parquet`; see Data shards below).
- [ ] **YAML data section:** `data_configs[0].on_disk_shard_root` points at that parent directory; `on_disk_data_sources` lists the **exact subdirectory names** to train on (`data_storage_type: on_disk` as needed).
- [ ] **YAML model / trainer:** Under `model_configs[0].model_args` and `lightning_trainer_pars`, sanity-check architecture, LR, `batch_size`, `num_workers`, checkpoint cadence, and **`num_nodes` must equal `run.sh` `--nodes`**.
- [ ] **`max_epochs`** (and resume fields if used) align with how long you want to train.

## Environment

Mamba env at `~/deg_venv` (path-based, not a conda *name*). `run.sh` activates
it automatically — see **`run.sh` ~line 120 (`mamba activate ~/deg_venv`)**.
Edit that path if your env lives elsewhere.

To activate manually (e.g. to run Python locally):

```bash
mamba activate ~/deg_venv
cd ~/lpm_style
poetry run python -c 'import perturb_lib; print("ok")'
```

Refresh deps after `pyproject.toml` changes: `poetry install`.

To create the env from scratch (e.g. on a new machine), use **`--prefix`** so
it matches `run.sh` and `mamba activate ~/deg_venv`:

```bash
mamba env create --prefix ~/deg_venv --file env.yml
mamba activate ~/deg_venv
poetry install                # installs the perturb_lib deps from pyproject.toml
```

If you use a different directory, pass that same path to `--prefix`/`activate`
and update the `mamba activate ...` line in `run.sh` (~line 120) to match.

## Submitting

```bash
sbatch run.sh                  # default: cleans .plib_cache/results/lpm_modified first
CLEAN_RESULTS=0 sbatch run.sh  # disable pre-clean (e.g. when resuming)
squeue --me                    # check queue
scancel <jobid>                # cancel
```

Slurm knobs in `run.sh` `#SBATCH` block: **`--nodes`**, **`--gpus-per-task`**, **`--ntasks-per-node`**, **`--cpus-per-task`**, **`--mem`**, **`-t` (walltime)**, **`--qos`**, **`--partition`**, **`--exclude`**. **`--nodes` must match `num_nodes` in the YAML** (`model_args.lightning_trainer_pars`).

Training entry point:

```bash
poetry run python -m perturb_gym.training train_from_config_file \
  --config_file_id_or_path=<id>    # mirrored at the bottom of run.sh (e.g. lpm_modified)
```

For routine experiments you usually only edit the YAML and re-`sbatch`; no code changes required.

## Data shards

`data_configs[0].on_disk_shard_root` is the parent directory containing one
folder per dataset; `on_disk_data_sources` selects which subfolders to use.
Typically many datasets live under `.plib_cache/plibdata/`, produced by the
`LPM_style_step1/2/3_*.ipynb` notebooks.

Each dataset folder has the same shape:

```
.plib_cache/plibdata/dili_train_CL_0000182/
├── info.json              # {"PDATA_FORMAT_VERSION": 1, "SHARDSIZE": 200000, ...}
├── metadata.parquet       # row-level metadata (split labels, timestamps, etc.)
└── shard_NNNNNN.parquet   # 200k rows each, ~2.3 MB; data is split across N shards
```

To add or remove a dataset from training, edit the `on_disk_data_sources` list
in the YAML — the entries are exactly the directory names under
`on_disk_shard_root`. To regenerate the shards from raw inputs, rerun the
`LPM_style_step*` notebooks.

## Config

YAML files live under `perturb_gym/configs/collection/`. The stem of the filename
(without `.yaml`) is the id you pass as `--config_file_id_or_path` (e.g. `lpm_modified`).

Three main sections:

- `environment_configs` — seeds  
- `data_configs` — shard root + which folders (`on_disk_data_sources`)  
- `model_configs` — model hyperparameters + `lightning_trainer_pars` (DDP, epochs, …)

Knobs you'll most often touch, under `model_configs[0].model_args`:

| Key | Purpose | Typical |
|---|---|---|
| `batch_size` | Per-rank micro-batch. Global = `batch_size × num_nodes`. | 4096–16384 |
| `learning_rate` | Initial LR. | 1e-3 to 5e-3 |
| `learning_rate_decay` | Per-epoch ExpLR factor. | 0.97 |
| `embedding_dim` / `hidden_dim` / `num_layers` / `dropout` | Architecture. | 128 / 256 / 2 / 0.1 |
| `num_workers` | DataLoader workers per rank. **6 with `on_disk`, 0 with `in_memory`.** | 6 |
| `epoch_checkpoint_every_n` | Save a Lightning `.ckpt` every Nth epoch. Set to `1` for every epoch or `0` to disable scheduled snapshots. | `1` (every epoch; lighter I/O: use `5`, `10`, …) |
| `epoch_checkpoint_save_last` | Roll a `last.ckpt` at every save event (same cadence as `every_n`). Used for `resume_from_checkpoint` after preemption. Set `false` to skip. | `true` |
| `resume_from_checkpoint` | Absolute path to a `.ckpt`, or remove for fresh. | `null` |

Under `model_args.lightning_trainer_pars`:

| Key | Purpose |
|---|---|
| `max_epochs` | Training ceiling. **Must exceed saved epoch when resuming.** |
| `num_nodes` | Must equal `--nodes` in `run.sh`. |

## Checkpoints

All outputs per run live under
`.plib_cache/results/lpm_modified/<model_hash>/seed_<seed>/`:

- **`model.pt`** — weights only, written once at end of training. For inference / fine-tuning.
  **Cannot be passed to `resume_from_checkpoint`** (no optimizer state).
- **`checkpoints/epoch-NNNN.ckpt`** — full Lightning snapshots (weights +
  optimizer + LR scheduler + step + epoch + RNG) at the cadence set by
  `epoch_checkpoint_every_n`, never overwritten. Use these for
  `resume_from_checkpoint`. Weights for inference are accessible via
  `torch.load(path)["state_dict"]`.
- **`checkpoints/last.ckpt`** — rolling latest, overwritten at every save
  event when `epoch_checkpoint_save_last: true` (same cadence as
  `every_n`). Always at an epoch boundary, so safe for
  `resume_from_checkpoint`. Intended for preemption / crash recovery.

Scheduled-save epochs = multiples of `epoch_checkpoint_every_n`. With the
current YAML (`every_n: 1`, `max_epochs: 25`): `{1, 2, …, 25}` — a full
checkpoint at the end of **every** epoch. Use a larger `every_n` (e.g. `5`) if
you want fewer files and less disk use.

## Layout

| Path | Purpose |
|---|---|
| `run.sh` | Slurm batch script. `sbatch run.sh` launches everything. |
| `perturb_gym/configs/collection/` | YAML training configs (`<id>.yaml`; id used in `--config_file_id_or_path`). |
| `perturb_gym/training.py` | Entry point invoked by `run.sh`. |
| `perturb_lib/models/collection/lpm.py` | LPM model + checkpoint callbacks. |
| `.plib_cache/plibdata/` | Pre-built parquet shards (one folder per dataset). |
| `.plib_cache/results/lpm_modified/` | Output: TensorBoard logs, `model.pt`, checkpoints. |
| `perturb_lib.<jobid>.{out,err}` | Slurm log files. |
| `quick_train_dili.ipynb` | Single-GPU canary for fast smoke tests. |

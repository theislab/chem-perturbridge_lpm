# LPM-Style — Quick Reference

This is a modified fork of [perturblib/perturblib](https://github.com/perturblib/perturblib),
wired for multi-node DDP training of the 6-feature LPM on Slurm. The original library
README is preserved here as [README_original.md](README_original.md).

## Quick start

```bash
sbatch run.sh                  # train from scratch with current YAML
tail -f perturb_lib.<jobid>.out
```

Edit `perturb_gym/configs/collection/lpm_modified.yaml` to change anything about
training. Then resubmit. No code changes needed for routine experiments.

## Layout

| Path | Purpose |
|---|---|
| `run.sh` | Slurm batch script. `sbatch run.sh` launches everything. |
| `perturb_gym/configs/collection/lpm_modified.yaml` | **The config.** Edit this. |
| `perturb_gym/training.py` | Entry point invoked by `run.sh`. |
| `perturb_lib/models/collection/lpm.py` | LPM model + checkpoint callback. |
| `.plib_cache/plibdata/` | Pre-built parquet shards (one folder per dataset). |
| `.plib_cache/results/lpm_modified/` | Output: TensorBoard logs, `model.pt`, checkpoints. |
| `perturb_lib.<jobid>.{out,err}` | Slurm log files. |
| `quick_train_dili.ipynb` | Single-GPU canary for fast smoke tests. |

## Data shards

`data_configs[0].on_disk_shard_root` is the parent directory containing one
folder per dataset; `on_disk_data_sources` selects which subfolders to use.
Currently 160 datasets exist under `.plib_cache/plibdata/`, produced by the
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

## Environment

Mamba env at `~/deg_venv`. `run.sh` activates it automatically. To activate
manually (e.g. for a notebook):

```bash
mamba activate ~/deg_venv
cd ~/lpm_style
poetry run jupyter lab        # or any python ... command
```

Refresh deps after `pyproject.toml` changes: `poetry install`.

## Editing the config

`lpm_modified.yaml` has three sections: `environment_configs` (seeds),
`data_configs` (which shards to use), `model_configs` (model + trainer params).

The knobs you'll most often touch, all under `model_configs[0].model_args`:

| Key | Purpose | Typical |
|---|---|---|
| `batch_size` | Per-rank micro-batch. Global = `batch_size × num_nodes`. | 4096–16384 |
| `learning_rate` | Initial LR. | 1e-3 to 5e-3 |
| `learning_rate_decay` | Per-epoch ExpLR factor. | 0.97 |
| `embedding_dim` / `hidden_dim` / `num_layers` / `dropout` | Architecture. | 128 / 256 / 2 / 0.1 |
| `num_workers` | DataLoader workers per rank. **6 with `on_disk`, 0 with `in_memory`.** | 6 |
| `epoch_checkpoint_special` | Specific epochs to checkpoint (1-indexed). Must be **wrapped in an outer list** because the config loader treats inner lists as grid-search sweeps. | `[[1]]` for just epoch 1; `[[1, 3]]` for epochs 1 and 3 |
| `epoch_checkpoint_every_n` | Save every Nth epoch. | `5` (→ epochs 5, 10, 15…) |
| `epoch_checkpoint_save_last` | Roll a `last.ckpt` + `last.pt` every epoch (epoch-boundary, safe to resume). For preemption recovery. Set `false` to skip the per-epoch I/O. | `true` |
| `resume_from_checkpoint` | Absolute path to a `.ckpt`, or remove for fresh. | `null` |

Under `model_args.lightning_trainer_pars`:

| Key | Purpose |
|---|---|
| `max_epochs` | Training ceiling. **Must exceed saved epoch when resuming.** |
| `num_nodes` | Must equal `--nodes` in `run.sh`. |

## Submitting

```bash
sbatch run.sh                  # default: cleans .plib_cache/results/lpm_modified first
CLEAN_RESULTS=0 sbatch run.sh  # disable pre-clean (e.g. when resuming)
squeue --me                    # check queue
scancel <jobid>                # cancel
```

Slurm knobs in `run.sh` `#SBATCH` block: `--nodes`, `--mem`, `--qos`,
`--exclude`, `-t` (walltime). **`--nodes` and YAML `num_nodes` must match.**

## Checkpoints

All outputs per run live under
`.plib_cache/results/lpm_modified/<model_hash>/seed_<seed>/`:

- **`model.pt`** — weights only, written once at end of training. For inference / fine-tuning.
  **Cannot be passed to `resume_from_checkpoint`** (no optimizer state).
- **`checkpoints/epoch-NNNN.ckpt`** — full Lightning snapshots (weights +
  optimizer + LR scheduler + step + epoch + RNG) at scheduled epochs only,
  never overwritten. Use these for `resume_from_checkpoint`.
- **`checkpoints/epoch-NNNN.pt`** — weights-only sibling of each `.ckpt` at
  scheduled epochs. Same format as the final `model.pt`; ~10× smaller than
  the matching `.ckpt`. Use for inference / probing intermediate epochs.
- **`checkpoints/last.ckpt`** + **`last.pt`** — rolling latest, overwritten at
  the end of every training epoch (when `epoch_checkpoint_save_last: true`).
  Always at an epoch boundary, so safe for `resume_from_checkpoint`. Intended
  for preemption / crash recovery between scheduled epochs.

Scheduled-save epochs = union of `epoch_checkpoint_special` and multiples of
`epoch_checkpoint_every_n`. With the current YAML: `{1, 5, 10, 15, …}`.

## Resuming from a checkpoint

1. Move the checkpoint dir **outside** `.plib_cache/results/lpm_modified/` so
   `run.sh`'s pre-clean doesn't delete it (e.g. `mv .plib_cache/results/lpm_modified .plib_cache/results/run1`).
2. In the YAML, set:
   ```yaml
         # any epoch-boundary .ckpt: a scheduled `epoch-NNNN.ckpt` or the rolling `last.ckpt`
         resume_from_checkpoint: /ictstr01/.../run1/.../checkpoints/last.ckpt
         ...
         max_epochs: 25                # > saved epoch, otherwise fit() exits immediately
   ```
3. `sbatch run.sh`. Look for `Resuming training from checkpoint: ...` in the log.

Use `last.ckpt` for the most-recent state (good for preemption recovery), or
`epoch-NNNN.ckpt` to rewind to a specific scheduled epoch. **Avoid mid-step
files** like the legacy `step-step=NNNNNNNN.ckpt`: Lightning's mid-step resume
is unreliable with multi-worker DDP DataLoaders.

Architecture must match — don't change `embedding_dim`, `hidden_dim`,
`num_layers`, or the dataset list between save and resume.

# LPM-Style — Quick Reference

Fork of [perturblib/perturblib](https://github.com/perturblib/perturblib) wired
for multi-node DDP training of the 6-feature LPM on Slurm. Original library
README preserved as [README_original.md](README_original.md).

---

## TL;DR

```bash
sbatch run.sh                   # train with the YAML named in run.sh
tail -f perturb_lib.<jobid>.out
```

If unsure, work top-to-bottom: **1. Environment** → **2. Submitting** →
**3. Data shards** → **4. Config** → **5. Checkpoints** → **6. Layout**.

---

## 1. Environment

The mamba env is **path-based** at `~/deg_venv` and activated by `run.sh`
(line 120: `mamba activate ~/deg_venv`).

### 1.1 Create the env (one-time)

```bash
mamba env create --prefix ~/deg_venv --file env.yml      # --prefix, NOT --name
mamba activate ~/deg_venv
poetry install                                           # installs perturb_lib deps
```

### 1.2 Activate manually (e.g. local Python)

```bash
mamba activate ~/deg_venv
cd ~/lpm_style
poetry run python -c 'import perturb_lib; print("ok")'
```

### 1.3 Use a different env path

If you put the env elsewhere, pass that path to **both** `--prefix`/`activate`
**and** edit `run.sh` line 120 to match. Refresh deps after editing
`pyproject.toml` with `poetry install`.

---

## 2. Submitting

### 2.1 Pre-flight checklist

Open `run.sh` and the YAML you intend to train, then verify:

| In `run.sh` | In the YAML | Must agree |
|---|---|---|
| `mamba activate ~/deg_venv` (line 120) | — | env path is correct |
| `#SBATCH --nodes=N` | `lightning_trainer_pars.num_nodes: N` | **same N** |
| `--config_file_id_or_path=<id>` (last line) | filename `<id>.yaml` exists | the right config |
| Slurm resources (`--gpus-per-task`, `--mem`, `-t`, `--qos`, `--partition`, `--exclude`) | — | fits your cluster |
| — | `data_configs[0].on_disk_shard_root` + `on_disk_data_sources` | shard folders exist (see §3) |
| — | `lightning_trainer_pars.max_epochs` | training length you want |

### 2.2 Submit / monitor / cancel

```bash
sbatch run.sh                   # submit (pre-cleans .plib_cache/results/lpm_modified)
CLEAN_RESULTS=0 sbatch run.sh   # submit without pre-clean (e.g. when resuming)
squeue --me                     # check queue
scancel <jobid>                 # cancel
tail -f perturb_lib.<jobid>.out
```

### 2.3 Direct entry point (without Slurm)

```bash
poetry run python -m perturb_gym.training train_from_config_file \
  --config_file_id_or_path=<id>
```

For routine experiments you only edit the YAML and re-`sbatch`; no code
changes required.

---

## 3. Data shards

### 3.1 Per-dataset folder layout

Each entry in `on_disk_data_sources` is a folder under `on_disk_shard_root`:

```
.plib_cache/plibdata/dili_train_CL_0000182/
├── info.json              # {"PDATA_FORMAT_VERSION": 1, "SHARDSIZE": 200000, ...}
├── metadata.parquet       # row-level metadata (split labels, timestamps, ...)
└── shard_NNNNNN.parquet   # 200k rows each, ~2.3 MB
```

### 3.2 Add / remove a dataset

1. Move/create its folder under `on_disk_shard_root`.
2. Add (or remove) the **exact** folder name in YAML `on_disk_data_sources`.
3. Re-`sbatch run.sh`.

### 3.3 Build shards from raw inputs

Rerun the `LPM_style_step1_*.ipynb`, `LPM_style_step2_*.ipynb`, and
`LPM_style_step3_*.ipynb` notebooks (in order).

---

## 4. Config

YAMLs live in `perturb_gym/configs/collection/`. The filename stem is the id
used in `--config_file_id_or_path` (e.g. `lpm_modified.yaml` → id
`lpm_modified`).

### 4.1 Sections

| Section | Purpose |
|---|---|
| `environment_configs` | Random seeds. |
| `data_configs` | Shard root + which folders to use (`on_disk_data_sources`). |
| `model_configs` | Architecture + optimizer + `lightning_trainer_pars`. |

### 4.2 Knobs in `model_configs[0].model_args`

| Key | Purpose | Typical |
|---|---|---|
| `batch_size` | Per-rank micro-batch. Global = `batch_size × num_nodes`. | 4096–16384 |
| `learning_rate` | Initial LR. | 1e-3 to 5e-3 |
| `learning_rate_decay` | Per-epoch ExpLR factor. | 0.97 |
| `embedding_dim` / `hidden_dim` / `num_layers` / `dropout` | Architecture. | 128 / 256 / 2 / 0.1 |
| `num_workers` | DataLoader workers per rank. **6 with `on_disk`, 0 with `in_memory`.** | 6 |
| `epoch_checkpoint_every_n` | Save `.ckpt` every Nth epoch. `0` disables, `1` saves every epoch. | `1` (use `5`/`10` for less I/O) |
| `epoch_checkpoint_save_last` | Roll a `last.ckpt` at the same cadence (preemption recovery). | `true` |
| `resume_from_checkpoint` | Absolute path to a `.ckpt`, or `null`. | `null` |

### 4.3 Knobs in `model_args.lightning_trainer_pars`

| Key | Purpose |
|---|---|
| `max_epochs` | Training ceiling. Must exceed the saved epoch when resuming. |
| `num_nodes` | **Must equal `#SBATCH --nodes` in `run.sh`.** |

---

## 5. Checkpoints

Per-run outputs live under
`.plib_cache/results/lpm_modified/<model_hash>/seed_<seed>/`.

### 5.1 What gets written

| File | Contents | Use for |
|---|---|---|
| `model.pt` | Weights only, written once at end of training. | Inference / fine-tuning. **Not** for resume. |
| `checkpoints/epoch-NNNN.ckpt` | Full Lightning snapshot (weights + optimizer + LR sched + step + epoch + RNG). Cadence = `epoch_checkpoint_every_n`. | `resume_from_checkpoint`. Weights via `torch.load(path)["state_dict"]`. |
| `checkpoints/last.ckpt` | Rolling latest, overwritten at every save event. | Preemption / crash recovery. |

### 5.2 Cadence

Saved epochs = multiples of `epoch_checkpoint_every_n`. With the current YAML
(`every_n: 1`, `max_epochs: 25`) → `{1, 2, …, 25}` (every epoch). Use a larger
`every_n` (e.g. `5`) for fewer files and less disk use.

---

## 6. Layout

| Path | Purpose |
|---|---|
| `run.sh` | Slurm batch script. `sbatch run.sh` launches everything. |
| `env.yml` | Conda env spec for `mamba env create`. |
| `pyproject.toml` | Python deps installed via `poetry install`. |
| `perturb_gym/configs/collection/` | YAML training configs (`<id>.yaml`). |
| `perturb_gym/training.py` | Entry point invoked by `run.sh`. |
| `perturb_lib/models/collection/lpm.py` | LPM model + checkpoint logic. |
| `.plib_cache/plibdata/` | Pre-built parquet shards (one folder per dataset). |
| `.plib_cache/results/<config_id>/` | Output: TensorBoard logs, `model.pt`, checkpoints. |
| `perturb_lib.<jobid>.{out,err}` | Slurm log files. |
| `quick_train_dili.ipynb` | Single-GPU canary for smoke tests. |

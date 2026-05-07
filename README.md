# LPM-inspired model trained on transcriptomic responses from the Chem-PerturBridge resource 

It is a fork of [perturblib/perturblib](https://github.com/perturblib/perturblib) adapted for multi-node DDP training of the 6-feature LPM (an MLP over 4 embeddings (`dataset`, `context`, `perturbation`, `readout`) and 2 continuous features (`log-dose`, `time`)) on a SLURM-managed HPC cluster. The original README is preserved as [README_original.md](README_original.md).

---

## Quick start guide

This repository runs LPM training on the Chem-PerturBridge dataset collection across distributed GPU nodes via SLURM. Two training modes are shipped: L1000-only and multi-dataset.

### 1. Environment

The Mamba environment must be set up once before training. The path `./lpm_training_venv` below is just an example; pick any location you like, but if you change it, update the `activate_env` function in `run.sh` to match.

**1.1** **Create the environment**
```
cd ~/lpm_style
mamba env create --prefix ./lpm_training_venv --file env.yml
mamba activate ./lpm_training_venv
poetry install
```

**1.2** **Verify the environment (optional)**
```
cd ~/lpm_style
mamba activate ./lpm_training_venv
poetry run python -c 'import perturb_lib; print("ok")'
```

**NB!** Re-run `poetry install` after editing `pyproject.toml`.

### 2. Data

Training data is read from pre-built parquet shards. The shard root is set by `data_configs[0].on_disk_shard_root` in the YAML config (default `.plib_cache/plibdata/`). The Chem-PerturBridge datasets are converted into this format via the `LPM_style_step{1,2,3}_*.ipynb` notebooks (see 2.3).

**2.1** **Per-dataset folder layout**

Each entry in `data_configs[0].on_disk_data_sources` is a folder under the shard root, for example `dili_train_CL_0000182`:
```
.plib_cache/plibdata/dili_train_CL_0000182/
├── info.json              # data format version and shard size (200000 rows)
├── metadata.parquet       # row-level metadata (split labels, timestamps, ...)
└── shard_NNNNNN.parquet   # 200000 rows per shard, about 2.3 MB
```

**2.2** **Add or remove a data slice**

Each entry is a `(dataset, context)` slice named `<dataset_id>_<context_id>` (e.g. `l1000_phase1_CVCL_0023`). Append a name to add a slice, delete one to remove it. The shard folder must exist under the shard root; otherwise rebuild it via the notebooks (2.3). Then re-submit with `sbatch run.sh -c <config_id>`.

**2.3** **Build shards from raw inputs**

Rerun the `LPM_style_step1_*.ipynb`, `LPM_style_step2_*.ipynb`, and `LPM_style_step3_*.ipynb` notebooks in this order.

### 3. Config files

YAML configs live in `perturb_gym/configs/collection/`. The filename stem is the config id passed to `run.sh -c <id>`. Each YAML declares `environment_configs` (seeds), `data_configs` (shard root and slices), and `model_configs` (architecture, optimizer, trainer parameters). Two configs are shipped:

**3.1** `lpm_modified_l1000_data`: L1000 phase 1 and phase 2 only. Single-platform, original LPM domain.

**3.2** `lpm_modified_all_data` (default): full Chem-PerturBridge collection (L1000 phase 1/2, CIGS MCE/TCM, DILI, GDPx2, etc.).

**NB!** Both configs use the same 6-feature LPM architecture and differ only in `data_configs[0].on_disk_data_sources`.

**3.3** **Common knobs to adjust**

Model and training hyperparameters can be set in the YAML config under `model_configs[0].model_args`:

* `batch_size`: per-rank micro-batch (global = `batch_size` * `num_nodes`). Set to 16384 to speed up training.
* `learning_rate`: initial LR. Default 2e-3.
* `learning_rate_decay`: per-epoch ExpLR factor. Default 0.97.
* `embedding_dim`, `hidden_dim`, `num_layers`, `dropout`: architecture. Used default parameters: 128, 256, 2, 0.1.
* `num_workers`: DataLoader workers per rank. Use 6 for `on_disk`, 0 for `in_memory`.
* `epoch_checkpoint_every_n`: save a checkpoint every Nth epoch (0 disables). Set to 1 (every epoch).
* `epoch_checkpoint_save_last`: when `true`, also rolls `last.ckpt` for preemption recovery. Set to `true`.
* `resume_from_checkpoint`: absolute path to a `.ckpt`, or `null`. Set to `null`.

Inside `model_configs[0].model_args.lightning_trainer_pars`:

* `max_epochs`: training ceiling. When resuming, must exceed the saved epoch.
* `num_nodes`: must equal `#SBATCH --nodes` in `run.sh`.

Cluster resources (`--nodes`, `--gpus-per-task`, `--mem`, `-t`, `--qos`, `--partition`, etc.) are declared as `#SBATCH` directives at the top of `run.sh`. Edit them there.

### 4. Running the job

**4.1** **Pre-flight checks**

Check the `#SBATCH` specifications at the top of `run.sh` against your cluster policy, and make sure `#SBATCH --nodes=N` equals `lightning_trainer_pars.num_nodes` in the YAML. Run `bash run.sh --help` to list available config ids.

**4.2** **Submit**
```
cd ~/lpm_style
sbatch run.sh -c lpm_modified_l1000_data    # L1000-only
sbatch run.sh                               # multi-dataset (default)
```

**NB!** By default `run.sh` deletes `.plib_cache/results/<config_id>/` before launching. To keep prior results, either move them aside (`mv .plib_cache/results/<config_id> .plib_cache/results/<config_id>_backup`) or prepend `CLEAN_RESULTS=0` to the submit command.

**4.3** **Monitor and cancel**
```
squeue --me                              # check the queue
scancel <jobid>                          # cancel a running job
tail -f logs/perturb_lib.<jobid>.out     # follow the training log
```

SLURM logs are written to `logs/perturb_lib.<jobid>.{out,err}` (path set by `LOG_DIR` in `run.sh`).

### 5. Checkpoints and outputs

Per-run outputs (checkpoints, `model.pt`, TensorBoard logs) live under `.plib_cache/results/<config_id>/<model_hash>/seed_<seed>/`.

**5.1** **What gets written**

* `model.pt`: weights-only, written at end of training. For inference or fine-tuning. Not for resume.
* `checkpoints/epoch-NNNN.ckpt`: full Lightning snapshots (weights, optimizer, LR scheduler, step/epoch, RNG). Frequency controlled by `epoch_checkpoint_every_n`. Use with `resume_from_checkpoint`, or load weights via `torch.load(path)["state_dict"]`.
* `checkpoints/last.ckpt`: rolling latest, overwritten at every save. For preemption or crash recovery.

**5.2** **Save frequency**

Checkpoints are saved at multiples of `epoch_checkpoint_every_n`. With the shipped defaults (`every_n: 1`, `max_epochs: 25`) a checkpoint is saved every epoch. Use a larger value (e.g. 5 or 10) for fewer files.

**5.3** **TensorBoard tracking**

Training metrics (loss, LR, throughput, validation scores) are written via Lightning's `TensorBoardLogger` to `.plib_cache/results/<config_id>/<model_hash>/seed_<seed>/learning_curves/version_<N>/`. Inspect them with:
```
cd ~/lpm_style
poetry run tensorboard --logdir .plib_cache/results/<config_id>
```
Then open `http://localhost:6006`. Pointing `--logdir` at the config results root lets TensorBoard compare seeds and `model_hash` runs side by side.

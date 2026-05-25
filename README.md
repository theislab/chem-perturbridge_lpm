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

### 6. Reproduce the 10-seed molecule-holdout table

This workflow generates the paper-style table with 10 seeds for each model family:

* `all_datasets`
* `finetune_frozen_molecule_embeddings`
* `scratch_target_only`
* `all_datasets_morgan_fixed`
* `finetune_morgan_fixed`
* `scratch_target_only_morgan_fixed`
* `all_datasets_morgan_learned`
* `finetune_morgan_learned_fixed_updated_embeddings`

The fixed seed list is `13, 17, 19, 23, 29, 31, 37, 41, 43, 47`. All runs use the cross-dataset molecule-holdout split baked into the preprocessed shard root. The split artifact is:

```
results/cross_dataset_molecule_holdout_split_all_data_plus_tahoe_novartis_op3/df_annot_split.parquet
```

The preprocessed shard root used by the configs is:

```
/lustre/groups/ml01/workspace/olga.novitskaia/lpm_style/.plib_cache/plibdata_multiout/lpm_multiout_all_data_plus_tahoe_novartis_op3_molholdout_h100_2x_bs4096_200epoch_lustre
```

The source and single-dataset base configs must exist before generating the 10-seed configs:

```
perturb_gym/configs/collection/lpm_multiout_transfer_source_all_data_plus_tahoe_novartis_op3_molholdout_h100_2x_bs4096_200epoch_lustre.yaml
perturb_gym/configs/collection/lpm_multiout_all_data_plus_tahoe_novartis_op3_molholdout_h100_2x_bs4096_200epoch_lustre.yaml
```

#### 6.1 Generate and submit the non-Morgan runs

Generate configs for the all-dataset source runs and target-only scratch runs:

```
./lpm_training_venv/bin/python scripts/create_lpm_paper10_configs.py
```

This writes `results/lpm_paper10_config_manifest.tsv` plus YAML configs under `perturb_gym/configs/collection/`.

Submit the all-dataset source and scratch target-only runs:

```
./lpm_training_venv/bin/python scripts/submit_lpm_paper10_source_scratch_jobs.py
```

This script submits:

* 10 all-dataset source jobs.
* 120 scratch target-only jobs: 12 datasets x 10 seeds.
* A dependent selector job, `run_lpm_paper10_select_and_submit.sh`, after the 10 source jobs finish.

The selector job chooses, for each target dataset, the all-dataset source checkpoint with the best validation RMSE across all source seeds and epochs. It then creates and submits:

* 120 `finetune_frozen_molecule_embeddings` jobs.
* 10 source-checkpoint eval jobs, one per source seed.
* A dependent summary job.

Useful outputs from this wave:

```
results/lpm_paper10_source_scratch_slurm_jobs.tsv
results/lpm_paper10_source_dataset_checkpoint_selection.tsv
results/lpm_paper10_source_dataset_checkpoint_selection_long.tsv
results/lpm_paper10_finetune_config_manifest.tsv
results/lpm_paper10_finetune_slurm_jobs.tsv
results/lpm_paper10_source_dataset_eval_seed*.tsv
```

#### 6.2 Build Morgan fingerprint embeddings

Build the frozen Morgan fingerprint artifact before creating Morgan configs:

```
./lpm_training_venv/bin/python scripts/build_morgan_perturbation_embeddings.py
```

By default this writes:

```
.plib_cache/morgan_perturbation_embeddings/pubchem_morgan_radius2_nbits128/
```

The produced `compound_embeddings.npy` is a 128-bit radius-2 Morgan fingerprint matrix aligned to the perturbation vocabulary. Missing or control entries are zero vectors unless `--missing-policy error` is used.

#### 6.3 Generate and submit Morgan runs

Generate configs for both Morgan variants:

```
./lpm_training_venv/bin/python scripts/create_lpm_paper10_morgan_variant_configs.py --variant all
```

This writes:

```
results/lpm_paper10_morgan_fixed_config_manifest.tsv
results/lpm_paper10_morgan_learned_config_manifest.tsv
```

Submit the frozen-Morgan runs:

```
./lpm_training_venv/bin/python scripts/submit_lpm_paper10_morgan_variant_jobs.py --variant morgan_fixed
```

This submits:

* 10 all-dataset runs with frozen Morgan molecule embeddings.
* 120 scratch target-only runs with frozen Morgan molecule embeddings.
* A dependent selector job that creates and submits 120 `finetune_morgan_fixed` jobs.
* Per-source eval jobs and a summary job.

Submit the Morgan-initialized-learned source runs:

```
./lpm_training_venv/bin/python scripts/submit_lpm_paper10_morgan_variant_jobs.py --variant morgan_learned
```

This submits:

* 10 all-dataset runs initialized from Morgan fingerprints, with molecule embeddings learnable during all-dataset training.
* A dependent selector job that creates and submits 120 `finetune_morgan_learned_fixed_updated_embeddings` jobs. These fine-tunes initialize from the selected all-dataset checkpoint and keep the updated molecule embeddings fixed.
* Per-source eval jobs and a summary job.

Useful outputs from the Morgan waves:

```
results/lpm_paper10_morgan_fixed_source_scratch_slurm_jobs.tsv
results/lpm_paper10_morgan_fixed_source_dataset_checkpoint_selection.tsv
results/lpm_paper10_morgan_fixed_finetune_config_manifest.tsv
results/lpm_paper10_morgan_fixed_finetune_slurm_jobs.tsv
results/lpm_paper10_morgan_fixed_source_dataset_eval_seed*.tsv

results/lpm_paper10_morgan_learned_source_scratch_slurm_jobs.tsv
results/lpm_paper10_morgan_learned_source_dataset_checkpoint_selection.tsv
results/lpm_paper10_morgan_learned_finetune_config_manifest.tsv
results/lpm_paper10_morgan_learned_finetune_slurm_jobs.tsv
results/lpm_paper10_morgan_learned_source_dataset_eval_seed*.tsv
```

#### 6.4 Monitor and regenerate the table

Monitor jobs with:

```
squeue --me
sacct -u "$USER" --starttime now-2days --format=JobID,JobName%80,State,Elapsed,Timelimit,ExitCode
```

The submit scripts create summary jobs automatically. You can also regenerate the table at any time from whatever results are present:

```
./lpm_training_venv/bin/python scripts/summarize_lpm_paper10_results.py
```

Final table outputs:

```
results/lpm_paper10_results_long.tsv
results/lpm_paper10_results_summary.tsv
results/lpm_paper10_results_summary.md
```

`results/lpm_paper10_results_long.tsv` contains one row per model family, seed, and dataset. `results/lpm_paper10_results_summary.tsv` contains the mean +/- standard deviation table used for paper reporting.

To write a separate intermediate snapshot instead of overwriting the default files:

```
./lpm_training_venv/bin/python scripts/summarize_lpm_paper10_results.py \
  --output-prefix results/lpm_paper10_results_current_check
```

#### 6.5 Extract best fine-tuned Morgan-learned embeddings

After `results/lpm_paper10_results_long.tsv` exists and the `finetune_morgan_learned_fixed_updated_embeddings` rows are complete, extract the best-validation checkpoint embeddings for each dataset:

```
./lpm_training_venv/bin/python scripts/extract_lpm_paper10_ft_morgan_learned_fixmol_embeddings.py \
  --long-table results/lpm_paper10_results_long.tsv \
  --output-root results/lpm_paper10_ft_morgan_learned_fixmol_best_embeddings
```

For each dataset, the extractor selects the `finetune_morgan_learned_fixed_updated_embeddings` checkpoint with the lowest validation RMSE, then writes line/context embeddings and molecule embeddings for all lines and molecules present in that dataset's configured shards, including validation and test molecules.

Main output:

```
results/lpm_paper10_ft_morgan_learned_fixmol_best_embeddings/best_checkpoint_embedding_exports.tsv
```

Each dataset also gets its own output directory:

```
results/lpm_paper10_ft_morgan_learned_fixmol_best_embeddings/<dataset_slug>/
|-- line_embeddings.npy
|-- line_metadata.parquet
|-- line_metadata.tsv
|-- df_line.pkl
|-- molecule_embeddings.npy
|-- molecule_metadata.parquet
|-- molecule_metadata.tsv
|-- df_molecule.pkl
`-- manifest.json
```

The metadata files align rows to embedding matrix rows. `df_line.pkl` and `df_molecule.pkl` are convenience pandas pickles with list-valued embedding columns.

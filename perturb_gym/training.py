"""Copyright (C) 2025  GlaxoSmithKline plc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Module used to train perturbation models.
"""

import base64
import json
import os
import shutil
import sys
import time
import tracemalloc
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from random import random
from typing import Any, Dict

import fire
import torch
import yaml

import perturb_lib as plib
from perturb_gym.configs.access import load_training_configs
from perturb_gym.configs.base import DataConfig, EnvironmentConfig, ModelConfig, TrainingConfig
from perturb_gym.evaluation import evaluate_model
from perturb_gym.paths import DEFAULT_RESULTS_DIRNAME, EVALUATION_LOG_FILENAME, TRAINING_LOG_FILENAME
from perturb_gym.utils import get_user_confirmation, hash_training_config_excluding_seed


def _to_yaml_safe(value):
    """Recursively coerce values to YAML-safe primitives.

    yaml.safe_dump only handles basic Python types (str, int, float, bool, None,
    list, tuple, dict). Anything else (e.g. pathlib.Path, set) raises
    RepresenterError. We walk dicts/lists and convert known offenders to strings
    so the training log can always be written.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_yaml_safe(v) for v in value]
    if isinstance(value, set):
        return [_to_yaml_safe(v) for v in value]
    return value


def _wandb_key_part(value: Any) -> str:
    return str(value).replace("/", "_")


def _flatten_numeric_metrics(prefix: str, value: Any) -> dict[str, float | int]:
    if isinstance(value, Mapping):
        metrics: dict[str, float | int] = {}
        for key, child_value in value.items():
            metrics.update(_flatten_numeric_metrics(f"{prefix}/{_wandb_key_part(key)}", child_value))
        return metrics
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {prefix: value}
    return {}


def _get_wandb_logger(logger_config: Any) -> Any | None:
    if isinstance(logger_config, (list, tuple)):
        for logger_obj in logger_config:
            wandb_logger = _get_wandb_logger(logger_obj)
            if wandb_logger is not None:
                return wandb_logger
        return None
    if logger_config is not None and logger_config.__class__.__name__ == "WandbLogger":
        return logger_config
    return None


def _wandb_update_config(wandb_logger: Any | None, values: Mapping[str, Any]) -> None:
    if wandb_logger is None:
        return
    try:
        wandb_logger.experiment.config.update(_to_yaml_safe(dict(values)), allow_val_change=True)
    except Exception as exc:
        plib.logger.warning(f"Could not update W&B config: {exc}")


def _wandb_log_metrics(wandb_logger: Any | None, metrics: Mapping[str, float | int]) -> None:
    if wandb_logger is None or not metrics:
        return
    try:
        wandb_logger.log_metrics(dict(metrics))
        for key, value in metrics.items():
            wandb_logger.experiment.summary[key] = value
    except Exception as exc:
        plib.logger.warning(f"Could not log metrics to W&B: {exc}")


def _wandb_save_file(wandb_logger: Any | None, path: Path) -> None:
    if wandb_logger is None or not path.exists():
        return
    try:
        wandb_logger.experiment.save(str(path), policy="now")
    except TypeError:
        wandb_logger.experiment.save(str(path))
    except Exception as exc:
        plib.logger.warning(f"Could not save {path} to W&B: {exc}")


def _is_global_zero_process() -> bool:
    for key in ("RANK", "LOCAL_RANK"):
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value) == 0
        except ValueError:
            continue
    return True


def _find_best_validation_checkpoint(model: Any) -> Path | None:
    checkpoint_path = getattr(model, "best_validation_checkpoint_path", None)
    if checkpoint_path:
        path = Path(checkpoint_path)
        if path.is_file():
            return path

    checkpoint_dir = getattr(model, "model_checkpoints_path", None)
    if checkpoint_dir is None:
        return None
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None
    candidates = sorted(
        checkpoint_dir.glob("best-validation-rmse*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_checkpoint_state(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
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
        metadata["model_args"] = _to_yaml_safe(model_args)
        return state, metadata

    raise ValueError(f"Unsupported checkpoint format: {path}")


def _extract_compound_embeddings_from_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    best_validation_rmse: float | None = None,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    state, metadata = _load_checkpoint_state(checkpoint_path)
    symbols = list(state["perturb_symbols"])
    embeddings = state["perturb_embedding_layer.weight"].detach().cpu().numpy().astype(np.float32)
    if len(symbols) != embeddings.shape[0]:
        raise ValueError(
            f"Symbol count ({len(symbols)}) does not match embedding rows ({embeddings.shape[0]})."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_df = pd.DataFrame({"code": np.arange(len(symbols), dtype=np.int32), "symbol": symbols})
    metadata_parquet = output_dir / "compound_metadata.parquet"
    metadata_tsv = output_dir / "compound_metadata.tsv"
    pickle_path = output_dir / "df_pert.pkl"
    manifest_path = output_dir / "manifest.json"

    metadata_df.to_parquet(metadata_parquet, index=False)
    metadata_df.to_csv(metadata_tsv, sep="\t", index=False)

    df_pert = metadata_df.copy()
    df_pert["lpm_style_embeddings"] = embeddings.tolist()
    df_pert.to_pickle(pickle_path)

    manifest = {
        **metadata,
        "best_validation_rmse": best_validation_rmse,
        "n_compounds": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "outputs": {
            "df_pert_pickle": str(pickle_path),
            "metadata_parquet": str(metadata_parquet),
            "metadata_tsv": str(metadata_tsv),
            "manifest_json": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(_to_yaml_safe(manifest), indent=2) + "\n")
    return manifest


def _wandb_finish(wandb_logger: Any | None) -> None:
    if wandb_logger is None:
        return
    try:
        wandb_logger.experiment.finish()
    except Exception as exc:
        plib.logger.warning(f"Could not finish W&B run cleanly: {exc}")


def _data_size_metrics(
    data_config: DataConfig,
    traindata: plib.PlibData,
    valdata: plib.PlibData | None,
    testdata: plib.PlibData | None,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    for split, data in (("train", traindata), ("val", valdata), ("test", testdata)):
        if data is None:
            continue
        try:
            metrics[f"data/{split}_rows"] = len(data)
        except Exception as exc:
            plib.logger.warning(f"Could not compute {split} data length for W&B: {exc}")
    data_sources = getattr(data_config, "on_disk_data_sources", None)
    if data_sources is not None:
        metrics["data/source_count"] = len(data_sources)
    return metrics


def train_from_args(training_config: TrainingConfig | Dict, results_dir: Path | str):
    """Train a single perturbation model given a training configuration.

    Args:
        training_config: training config object.
        results_dir: where to store the results of training.
    """
    # convert to TrainingConfig if not in that form already
    if not isinstance(training_config, TrainingConfig):
        training_config = TrainingConfig(
            environment_config=EnvironmentConfig(**training_config["environment_config"]),
            data_config=DataConfig(**training_config["data_config"]),
            model_config=ModelConfig(**training_config["model_config"]),
        )

    # resolve the results directory
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise ValueError(f"{results_dir} does not exit!")

    # fix random seed
    seed = training_config.environment_config.seed
    plib.set_all_seeds(seed if seed is not None else plib.get_seed())
    matmul_precision = os.environ.get("PERTURB_GYM_FLOAT32_MATMUL_PRECISION")
    if matmul_precision:
        torch.set_float32_matmul_precision(matmul_precision)
        plib.logger.info(f"Set float32 matmul precision to {matmul_precision}.")

    # start tracking memory consumption
    tracemalloc.start()

    # get train/val/test data
    traindata, valdata, testdata = training_config.get_train_val_test_data()

    # model naming and save directory preparation
    plib.logger.info(f"Loading and training model with id={training_config.model_config.model_id}")
    unique_model_name = (
        f"{training_config.model_config.model_id}_{hash_training_config_excluding_seed(training_config)}"
    )
    model_dir = results_dir / unique_model_name / f"seed_{seed}"
    model_dir.mkdir(exist_ok=True, parents=True)

    # model loading
    model = plib.load_model(training_config.model_config.model_id, training_config.model_config.model_args)

    # set up model training logging if specified and applicable
    wandb_logger = None
    wandb_run_config = {
        "results_dir": str(model_dir),
        "seed": seed,
        "data_config": dict(training_config.data_config),
        "model_config": {
            "model_id": training_config.model_config.model_id,
            "model_args": training_config.model_config.model_args,
            "save_model_after_training": training_config.model_config.save_model_after_training,
            "torch_compile": training_config.model_config.torch_compile,
            "resume_from_checkpoint": training_config.model_config.resume_from_checkpoint,
        },
    }
    if hasattr(model, "add_logger"):
        model.add_logger(
            model_dir,
            "learning_curves",
            wandb_config=training_config.model_config.wandb_config,
            run_config=_to_yaml_safe(wandb_run_config),
        )
        wandb_logger = _get_wandb_logger(getattr(model, "lightning_trainer_pars", {}).get("logger"))
    _wandb_update_config(wandb_logger, wandb_run_config)
    _wandb_log_metrics(wandb_logger, _data_size_metrics(training_config.data_config, traindata, valdata, testdata))

    # optional model compilation
    if training_config.model_config.torch_compile and isinstance(model, torch.nn.Module):
        plib.logger.info("Model compilation in the max-autotune mode..")
        model = torch.compile(model, "max-autotune")  # type: ignore[call-overload]

    # model training
    resume_ckpt = training_config.model_config.resume_from_checkpoint
    if resume_ckpt is not None:
        plib.logger.info(f"Resuming training from checkpoint: {resume_ckpt}")
    start_time = time.time()
    plib.logger.info("Model training..")
    model.fit(traindata=traindata, valdata=valdata, resume_from_checkpoint=resume_ckpt)
    training_time = time.time() - start_time
    plib.logger.info("Model training done!")
    tracemalloc.start()  # stop tracking memory consumption
    _, peak_memory = tracemalloc.get_traced_memory()

    if not _is_global_zero_process():
        plib.logger.info("Skipping post-training artifact export and evaluation on non-zero distributed rank.")
        tracemalloc.stop()
        _wandb_finish(wandb_logger)
        return

    # save the model if specified
    if training_config.model_config.save_model_after_training:
        plib.logger.info("Saving trained model...")
        plib.save_trained_model(model, model_dir / "model.pt", training_config.model_config.model_args)

    best_validation_checkpoint_path = _find_best_validation_checkpoint(model)
    best_validation_rmse = getattr(model, "best_validation_score", None)
    if best_validation_checkpoint_path is not None:
        _wandb_update_config(
            wandb_logger,
            {
                "best_validation_checkpoint": str(best_validation_checkpoint_path),
                "best_validation_rmse": best_validation_rmse,
            },
        )
        if best_validation_rmse is not None:
            _wandb_log_metrics(wandb_logger, {"checkpoint/best_validation_rmse": float(best_validation_rmse)})

    compound_embeddings_manifest = None
    if os.environ.get("PERTURB_GYM_EXTRACT_COMPOUND_EMBEDDINGS", "1") != "0":
        if best_validation_checkpoint_path is None:
            plib.logger.warning("Could not find a best Validation RMSE checkpoint; skipping compound embedding export.")
        else:
            embeddings_dir = model_dir / "compound_embeddings_best_validation_rmse"
            plib.logger.info(f"Extracting compound embeddings from {best_validation_checkpoint_path} to {embeddings_dir}")
            compound_embeddings_manifest = _extract_compound_embeddings_from_checkpoint(
                best_validation_checkpoint_path,
                embeddings_dir,
                None if best_validation_rmse is None else float(best_validation_rmse),
            )
            _wandb_log_metrics(
                wandb_logger,
                {
                    "compound_embeddings/n_compounds": compound_embeddings_manifest["n_compounds"],
                    "compound_embeddings/embedding_dim": compound_embeddings_manifest["embedding_dim"],
                },
            )
            _wandb_update_config(
                wandb_logger,
                {"compound_embeddings": compound_embeddings_manifest["outputs"]},
            )
            for output_path in compound_embeddings_manifest["outputs"].values():
                _wandb_save_file(wandb_logger, Path(output_path))

    # logging the training configuration
    plib.logger.info("Saving model training log...")
    model_training_log = {
        "data_config": dict(training_config.data_config),
        "seed": training_config.environment_config.seed,
        "model_id": training_config.model_config.model_id,
        "model_args": training_config.model_config.model_args,
        "torch_compile": training_config.model_config.torch_compile,
        "wandb_config": training_config.model_config.wandb_config,
        "best_validation_checkpoint": None
        if best_validation_checkpoint_path is None
        else str(best_validation_checkpoint_path),
        "best_validation_rmse": best_validation_rmse,
        "compound_embeddings": compound_embeddings_manifest,
        "training_time": f"{training_time / 60.0:.2f} min",
        "execution_datetime": datetime.now().strftime("%A, %B %d, %Y %I:%M:%S %p"),
        "peak_CPU-RAM_consumption": f"{peak_memory / (1024 * 1024):.1f} MB",
    }

    # extra log for pytorch models
    if isinstance(model, torch.nn.Module):
        if torch.cuda.is_available():
            device = "gpu"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        model_training_log["device"] = device
        if device == "gpu":
            model_training_log["gpu name"] = torch.cuda.get_device_properties(0).name
            model_training_log["max memory allocated per gpu"] = (
                f"{torch.cuda.max_memory_allocated(0) / (1024 ** 2):.1f} MB"
            )
            _wandb_log_metrics(
                wandb_logger,
                {"system/max_gpu_allocated_mb": torch.cuda.max_memory_allocated(0) / (1024**2)},
            )

    _wandb_log_metrics(
        wandb_logger,
        {
            "runtime/training_seconds": training_time,
            "runtime/training_minutes": training_time / 60.0,
            "system/peak_cpu_ram_mb": peak_memory / (1024 * 1024),
        },
    )

    training_log_path = model_dir / TRAINING_LOG_FILENAME
    with open(training_log_path, "w") as f:
        # DataConfig stores ``on_disk_shard_root`` as a pathlib.Path (see configs/base.py).
        # yaml.safe_dump only knows about basic Python types, so any Path leaking into the
        # log dict raises RepresenterError. Stringify recursively to keep the YAML readable.
        yaml.safe_dump(_to_yaml_safe(model_training_log), f, sort_keys=False)
    _wandb_save_file(wandb_logger, training_log_path)

    # model evaluation
    if os.environ.get("PERTURB_GYM_SKIP_EVAL") == "1":
        plib.logger.info("Skipping post-training evaluation because PERTURB_GYM_SKIP_EVAL=1.")
        _wandb_log_metrics(wandb_logger, {"final_eval/skipped": 1})
    else:
        plib.logger.info("Running post-training evaluation...")
        evaluation_results = evaluate_model(model, (traindata, valdata, testdata), model_dir)
        _wandb_log_metrics(wandb_logger, _flatten_numeric_metrics("final_eval", evaluation_results))
        _wandb_save_file(wandb_logger, model_dir / EVALUATION_LOG_FILENAME)

    # stop tracking memory consumption
    tracemalloc.stop()
    _wandb_finish(wandb_logger)
    plib.logger.info("All done!")


def train_from_b64_encoded_args(training_config: str, results_dir: Path | str):
    """Train a single perturbation model given a base64 encoded training configuration."""
    training_config_deserialized = json.loads(base64.b64decode(training_config))
    train_from_args(training_config_deserialized, results_dir)


def train_from_config_file(
    config_file_id_or_path: str,
    use_slurm: bool = False,
    experiment_probability: float = 1.0,
    slurm_args: str = "--mem=50G --time=12:00:00 --nodes=1 --partition=gpu --gres=gpu:1",
    results_dir: Path | str | None = None,
):
    """Train one or more perturbation models as specified in the configuration file.

    Args:
        config_file_id_or_path: Identifier of the config file within perturb-gym or custom path to one.
        use_slurm: Whether to run jobs in parallel on slurm.
        experiment_probability: probability of running each experiment. We stochastically decide whether to run the
        experiment based on this probability, essentially implementing random search to replace expensive grid search.
        slurm_args: If slurm is used, the arguments for the 'sbatch' call.
        results_dir: Where the results of training are to be stored.
    """
    if experiment_probability > 1.0 or experiment_probability < 0.0:
        raise ValueError("Experiment probability must be in [0,1]")

    # resolve the results directory
    results_dir = plib.get_path_to_cache() / DEFAULT_RESULTS_DIRNAME if results_dir is None else Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if config_file_id_or_path.endswith(".yaml"):
        config_results_dir = results_dir / Path(config_file_id_or_path).stem
    else:
        config_results_dir = results_dir / config_file_id_or_path

    if config_results_dir.is_dir() and any(config_results_dir.iterdir()):
        if os.environ.get("PERTURB_GYM_RESULTS_PRE_CLEANED") == "1":
            plib.logger.info(
                f"Continuing with existing results directory '{config_results_dir}' "
                "because PERTURB_GYM_RESULTS_PRE_CLEANED=1."
            )
        elif not get_user_confirmation(f"This will delete all files in '{config_results_dir}'. Continue? (yes/no)"):
            return
        else:
            shutil.rmtree(config_results_dir)
    if not config_results_dir.exists():
        config_results_dir.mkdir(exist_ok=True)

    # loop over training configurations training different perturbation models
    for training_config in load_training_configs(config_file_id_or_path):
        if random() > experiment_probability:
            continue
        if not use_slurm:  # run locally
            train_from_args(training_config, config_results_dir)
        else:  # send a job to slurm
            # serialize training_config
            training_config_serialized = base64.b64encode(json.dumps(training_config).encode("utf-8")).decode("utf-8")

            cli_command = f"{sys.executable} -m {__package__}.{Path(__file__).stem} train_from_b64_encoded_args"
            cli_command_with_args = (
                f"{cli_command} "
                f'--training_config "{training_config_serialized}" '
                f'--results_dir "{str(config_results_dir)}"'
            )
            model_args = training_config["model_config"]["model_args"]
            ncpus = 2 if ("num_workers" not in model_args) else (model_args["num_workers"] + 2)
            slurm_args += f" --cpus-per-task={ncpus}"
            slurm_command_with_args = f"sbatch {slurm_args} --wrap '{cli_command_with_args}'"
            os.system(slurm_command_with_args)


if __name__ == "__main__":
    fire.Fire()

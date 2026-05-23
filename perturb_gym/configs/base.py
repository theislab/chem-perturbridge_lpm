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

Training config-related modules.
"""

from pathlib import Path
from typing import Dict, List, Literal, Tuple, Type

import perturb_lib as plib
from perturb_lib import InMemoryPlibData, OnDiskPlibData, PlibData


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore


class EnvironmentConfig(dotdict):
    """Environment configuration data structure.

    Attributes:
        seed: Random seed to fix in all used libraries that have randomization.
    """

    def __init__(self, seed: int):
        super().__init__()
        self.seed = seed


class DataConfig(dotdict):
    """Data configuration data structure.

    Attributes:
        training_contexts: defines contexts available for training. It can be:
          1. a list of strings specifying the list of contexts to select
          2. a string that specifies the context to be selected
          3. a string that specifies the regular expression in which case all contexts that match it are selected
        on_disk_shard_root: Absolute path to a directory that contains pre-built shard folders (each with
            ``metadata.parquet``). Mutually exclusive with ``training_contexts`` when both would load data.
        on_disk_data_sources: Names of immediate subdirectories under ``on_disk_shard_root`` to stack together.
            Requires ``data_storage_type: on_disk``.
        on_disk_split_annotation_path: Optional raw sample split annotation used by preprocessing scripts when
            generating ``on_disk_shard_root``. The split is baked into shards and is not read during training.
        val_perturbations_selected_from: if specified, defines context from which validation perturbations are selected.
        val_and_test_perturbations_selected_from: if specified, defines context from which both validation and
        test perturbations are selected.
        cache_shards_in_memory: if true for on-disk prebuilt shards, DataLoader workers cache shards after first read.
        preprocessing_type: the type of preprocessing to apply on each context.
    """

    def __init__(
        self,
        training_contexts: List[str] | str | None = None,
        on_disk_shard_root: str | Path | None = None,
        on_disk_data_sources: List[str] | None = None,
        on_disk_split_annotation_path: str | Path | None = None,
        val_perturbations_selected_from: str | None = None,
        val_and_test_perturbations_selected_from: str | None = None,
        cache_shards_in_memory: bool = False,
        preprocessing_type: plib.PreprocessingType = plib.DEFAULT_PREPROCESSING_TYPE,
        data_storage_type: Literal["in_memory", "on_disk"] = "in_memory",
    ):
        super().__init__()
        uses_root = on_disk_shard_root is not None or on_disk_data_sources is not None
        if uses_root:
            if training_contexts is not None:
                raise ValueError("Use either training_contexts or on_disk_shard_root + on_disk_data_sources, not both.")
            if on_disk_shard_root is None or on_disk_data_sources is None:
                raise ValueError("on_disk_shard_root and on_disk_data_sources must be set together.")
            if data_storage_type not in ("on_disk", "in_memory"):
                raise ValueError(
                    "on_disk_shard_root requires data_storage_type: on_disk or in_memory."
                )
            self.training_contexts = None
            self.on_disk_shard_root = Path(on_disk_shard_root)
            self.on_disk_data_sources = list(on_disk_data_sources)
            if on_disk_split_annotation_path is not None:
                self.on_disk_split_annotation_path = Path(on_disk_split_annotation_path)
        else:
            if training_contexts is None:
                raise ValueError("Either training_contexts or on_disk_shard_root + on_disk_data_sources is required.")
            self.training_contexts = training_contexts
        if val_perturbations_selected_from is not None:
            self.val_perturbations_selected_from = val_perturbations_selected_from
        if val_and_test_perturbations_selected_from is not None:
            self.val_and_test_perturbations_selected_from = val_and_test_perturbations_selected_from
        self.cache_shards_in_memory = cache_shards_in_memory
        self.preprocessing_type = preprocessing_type
        self.data_storage_type = data_storage_type

    def get_train_val_test_data(self) -> Tuple[PlibData, PlibData | None, PlibData | None]:
        """Generating train/val/test data."""
        plibdata_type: Type[PlibData]
        if self.data_storage_type == "in_memory":
            plibdata_type = InMemoryPlibData
        elif self.data_storage_type == "on_disk":
            plibdata_type = OnDiskPlibData
        else:
            raise ValueError(f"Unrecognized data storage type: {self.data_storage_type}")

        if self.training_contexts is not None:
            all_data = plib.load_plibdata(
                self.training_contexts, preprocessing_type=self.preprocessing_type, plibdata_type=plibdata_type
            )
        else:
            assert self.on_disk_shard_root is not None and self.on_disk_data_sources is not None
            kwargs = {
                "data_sources": self.on_disk_data_sources,
                "path_to_data_sources": self.on_disk_shard_root,
            }
            if plibdata_type is OnDiskPlibData:
                kwargs["cache_shards_in_memory"] = self.cache_shards_in_memory
            all_data = plibdata_type(**kwargs)

        if "val_and_test_perturbations_selected_from" in self:  # validation and test data are specified
            vt = self.val_and_test_perturbations_selected_from
            context_ids = None if isinstance(vt, str) and vt.lower() == "all" else vt
            traindata, valdata, testdata = plib.split_plibdata_3fold(all_data, context_ids=context_ids)

        elif "val_perturbations_selected_from" in self:  # only validation data are specified
            if self.val_perturbations_selected_from.lower() == "all":
                context_ids = None
            else:
                context_ids = self.val_perturbations_selected_from
            traindata, valdata = plib.split_plibdata_2fold(all_data, context_ids=context_ids)
            testdata = None

        else:  # no validation nor test data
            traindata, valdata, testdata = all_data, None, None

        return traindata, valdata, testdata


class ModelConfig(dotdict):
    """Model configuration data structure.

    Attributes:
        model_id: Identifier of the model.
        model_args: Model arguments in the form of a dictionary.
        save_model_after_training: A flag to indicate whether to save the model after training.
        torch_compile: A flag to indicate whether to apply PyTorch compilation.
        resume_from_checkpoint: Path to a PyTorch Lightning ``.ckpt`` file to resume training from.
            When set, training continues from the given checkpoint (weights, optimizer state, and
            epoch/step counter are all restored). Requires ``enable_checkpointing: true`` in
            ``lightning_trainer_pars`` so that periodic checkpoints are written.
        wandb_config: Optional W&B logger settings. When ``enabled`` is true, these settings are
            passed to PyTorch Lightning's ``WandbLogger``.
    """

    def __init__(
        self,
        model_id: str,
        model_args: Dict | None = None,
        save_model_after_training: bool = False,
        torch_compile: bool = False,
        resume_from_checkpoint: str | None = None,
        wandb_config: Dict | None = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.model_args = model_args
        self.save_model_after_training = save_model_after_training
        self.torch_compile = torch_compile
        self.resume_from_checkpoint = resume_from_checkpoint
        self.wandb_config = wandb_config or {}


class TrainingConfig(dotdict):
    """Main class for defining training configs."""

    def __init__(
        self,
        environment_config: EnvironmentConfig,
        data_config: DataConfig,
        model_config: ModelConfig,
    ):
        super().__init__()
        self.environment_config = environment_config
        self.data_config = data_config
        self.model_config = model_config

    def get_train_val_test_data(self) -> Tuple[PlibData, PlibData | None, PlibData | None]:
        """Generating train/val/test data."""
        return self.data_config.get_train_val_test_data()

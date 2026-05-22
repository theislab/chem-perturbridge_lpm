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

Large perturbation model implementation.
"""

import string
from abc import ABCMeta
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl
import pytorch_lightning as pyl
import torch
from numpy.random import RandomState
from numpy.typing import NDArray
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn as nn
from tqdm import tqdm

from perturb_lib.data.access import Vocabulary, encode_data
from perturb_lib.data.plibdata import PlibData
from perturb_lib.environment import get_path_to_cache, logger
from perturb_lib.models._utils import LPMProfiler
from perturb_lib.models.access import register_model
from perturb_lib.models.base import ModelMixin, to_tensor_dict
from perturb_lib.utils import inherit_docstring


@inherit_docstring
@register_model
class LPM(ModelMixin, pyl.LightningModule, metaclass=ABCMeta):
    """Large perturbation model.

    Args:
        embedding_dim: Dimensionality of all embedding layers.
        optimizer_name: Name of pytorch optimizer to use.
        learning_rate: Learning rate.
        learning_rate_decay: Exponential learning rate decay.
        num_layers: Depth of the MLP.
        hidden_dim: Number of units in the hidden nodes.
        batch_size: Size of batches during training.
        embedding_aggregation_mode: Defines how to aggregate embeddings.
        num_workers: Number of workers to use during data loading.
        pin_memory: Whether to pin the memory.
        early_stopping_patience: Patience for early stopping in case validation set is given.
        lightning_trainer_pars: Parameters for pytorch-lightning.
        resume_from_checkpoint: Path to a ``.ckpt`` file written by PyTorch Lightning.
            When provided, ``trainer.fit`` resumes from that checkpoint (weights, optimizer
            state, and epoch/step counter are all restored). Requires
            ``enable_checkpointing: true`` in ``lightning_trainer_pars`` so that periodic
            checkpoints are actually written to disk during training.
        epoch_checkpoint_every_n: Save a Lightning ``.ckpt`` snapshot every Nth epoch
            (e.g. 5 -> epochs 5, 10, 15, ...). Set to 1 to save every epoch (incl. epoch 1)
            or 0 to disable scheduled snapshots. Snapshots are kept forever; callers are
            responsible for cleaning up old checkpoints if disk usage matters.
        epoch_checkpoint_save_last: If True, also write a rolling ``last.ckpt`` at every
            save event (i.e. every Nth epoch). Used for ``resume_from_checkpoint`` after
            preemption / crash. Set False to skip and save a bit of disk I/O.
        output_mode: ``scalar`` keeps the original one-row-per-readout behavior; ``multiout``
            predicts all readouts for each sample and computes loss only for available readouts.
            ``auto`` selects ``multiout`` when the dataset has sample-level ragged readout columns.
    """

    def __init__(
        self,
        embedding_dim: int,
        optimizer_name: str,
        learning_rate: float,
        learning_rate_decay: float,
        num_layers: int,
        hidden_dim: int,
        batch_size: int,
        embedding_aggregation_mode: Literal["sum", "mean", "max"] = "mean",
        dropout: float = 0.0,
        num_workers: int = 0,
        pin_memory: bool = True,
        early_stopping_patience: int = 0,
        profiler: bool = False,
        lightning_trainer_pars: dict | None = None,
        resume_from_checkpoint: str | None = None,
        epoch_checkpoint_every_n: int = 0,
        epoch_checkpoint_save_last: bool = True,
        output_mode: Literal["auto", "scalar", "multiout"] = "auto",
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.early_stopping_patience = early_stopping_patience
        # Epoch-based checkpointing is delegated to the stock Lightning ModelCheckpoint
        # callback (configured in `configure_callbacks`). Periodic snapshots are saved
        # every `epoch_checkpoint_every_n` epochs and kept forever; `last.ckpt` rolls at
        # the same cadence when `epoch_checkpoint_save_last=True`. Set every_n=1 to
        # capture every epoch (incl. epoch 1).
        self.epoch_checkpoint_every_n = epoch_checkpoint_every_n
        self.epoch_checkpoint_save_last = epoch_checkpoint_save_last
        self.optimizer_name = optimizer_name
        self.embedding_aggregation_mode = embedding_aggregation_mode
        self.output_mode = output_mode
        self.active_output_mode: Literal["scalar", "multiout"] = "scalar"
        self.output_dim = 1
        self.lightning_trainer_pars = {} if (lightning_trainer_pars is None) else lightning_trainer_pars
        self.resume_from_checkpoint: str | None = resume_from_checkpoint
        self.loss = nn.MSELoss(reduction="none")
        self.default_root_dir = get_path_to_cache()

        # vocabulary, to be initialized upon "fit"
        self.vocab: Vocabulary | None = None

        # actual embeddings, to be initialized upon "fit"
        self.dataset_embedding_layer: nn.Embedding | None = None
        self.context_embedding_layer: nn.Embedding | None = None
        self.perturb_embedding_layer: nn.EmbeddingBag | None = None
        self.readout_embedding_layer: nn.Embedding | None = None
        self.dataset_output_weight: nn.Parameter | None = None
        self.dataset_output_bias: nn.Parameter | None = None
        # Continuous-feature projections: a single scalar -> embedding_dim vector.
        self.log_dose_layer: nn.Linear | None = None
        self.time_layer: nn.Linear | None = None

        # prediction neural network
        self.predictor = self.build_predictor()

        # PL-related
        self.training_loss_sum: torch.Tensor | None = None
        self.training_loss_count: torch.Tensor | None = None
        self.validation_loss_sum: torch.Tensor | None = None
        self.validation_loss_count: torch.Tensor | None = None
        self.throughput_outputs: list[float] = []
        self.lightning_trainer_pars["default_root_dir"] = self.default_root_dir
        self.save_hyperparameters(ignore=["lightning_trainer_pars", "resume_from_checkpoint"])
        self.model_checkpoints_path = self.default_root_dir / "checkpoints"
        if profiler:
            self.lightning_trainer_pars["profiler"] = LPMProfiler()
        self.ckpt_filename: str | None
        if early_stopping_patience > 0:
            self.ckpt_filename = "".join(RandomState(None).choice(list(string.ascii_lowercase)) for _ in range(10))
        else:
            self.ckpt_filename = None

    def state_dict(self, *args, **kwargs):  # noqa: D102
        state = super().state_dict(*args, **kwargs)
        # ensure that the vocabulary is a part of model's state, so it gets serialized and saved when required
        if self.vocab is not None:
            state["context_symbols"] = self.vocab.context_vocab["symbol"].to_list()
            state["perturb_symbols"] = self.vocab.perturb_vocab["symbol"].to_list()
            state["readout_symbols"] = self.vocab.readout_vocab["symbol"].to_list()
            state["dataset_symbols"] = self.vocab.dataset_vocab["symbol"].to_list()
            state["active_output_mode"] = self.active_output_mode
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):  # noqa: D102
        state_dict = cast(dict[str, Any], state_dict)  # to avoid MyPy complaining about "pop" operation
        active_output_mode = state_dict.pop("active_output_mode", None)
        if "context_symbols" in state_dict:
            self.vocab = Vocabulary.initialize_from_symbols(
                context_symbols=state_dict["context_symbols"],
                perturb_symbols=state_dict["perturb_symbols"],
                readout_symbols=state_dict["readout_symbols"],
                dataset_symbols=state_dict["dataset_symbols"],
            )
            state_dict.pop("context_symbols")
            state_dict.pop("perturb_symbols")
            state_dict.pop("readout_symbols")
            state_dict.pop("dataset_symbols")
            if active_output_mode in {"scalar", "multiout"}:
                self.active_output_mode = active_output_mode
                self.output_dim = len(self.vocab.readout_vocab) if self.active_output_mode == "multiout" else 1
                self.predictor = self.build_predictor()
                if self.active_output_mode == "multiout" and "dataset_output_weight" in state_dict:
                    self._initialize_dataset_output_heads(len(self.vocab.dataset_vocab), len(self.vocab.readout_vocab))
        if "context_embedding_layer.weight" in state_dict:
            self.context_embedding_layer = nn.Embedding.from_pretrained(state_dict["context_embedding_layer.weight"])
        if "perturb_embedding_layer.weight" in state_dict:
            self.perturb_embedding_layer = nn.EmbeddingBag.from_pretrained(state_dict["perturb_embedding_layer.weight"])
        if "readout_embedding_layer.weight" in state_dict:
            self.readout_embedding_layer = nn.Embedding.from_pretrained(state_dict["readout_embedding_layer.weight"])
        if "dataset_embedding_layer.weight" in state_dict:
            self.dataset_embedding_layer = nn.Embedding.from_pretrained(state_dict["dataset_embedding_layer.weight"])
        if "log_dose_layer.weight" in state_dict and self.log_dose_layer is None:
            self.log_dose_layer = nn.Linear(1, self.embedding_dim)
        if "time_layer.weight" in state_dict and self.time_layer is None:
            self.time_layer = nn.Linear(1, self.embedding_dim)
        super().load_state_dict(state_dict, strict, assign)

    @staticmethod
    def _has_multiout_columns(columns: list[str]) -> bool:
        return {"dataset_code", "context_code", "perturbation_codes", "readout_codes"}.issubset(columns)

    @classmethod
    def _is_multiout_data(cls, data: PlibData[pl.DataFrame]) -> bool:
        return cls._has_multiout_columns(data.columns)

    def _resolve_output_mode(self, data: PlibData[pl.DataFrame]) -> Literal["scalar", "multiout"]:
        if self.output_mode == "auto":
            return "multiout" if self._is_multiout_data(data) else "scalar"
        return self.output_mode

    def _initialize_vocabularies_and_embeddings(self, data: PlibData[pl.DataFrame]):
        self.vocab = Vocabulary.initialize_from_data(data)
        self.active_output_mode = self._resolve_output_mode(data)
        self.output_dim = len(self.vocab.readout_vocab) if self.active_output_mode == "multiout" else 1

        self.dataset_embedding_layer = (
            None
            if self.active_output_mode == "multiout"
            else nn.Embedding(len(self.vocab.dataset_vocab), self.embedding_dim)
        )
        self.context_embedding_layer = nn.Embedding(len(self.vocab.context_vocab), self.embedding_dim)
        self.perturb_embedding_layer = nn.EmbeddingBag(
            len(self.vocab.perturb_vocab), self.embedding_dim, mode=self.embedding_aggregation_mode
        )
        self.readout_embedding_layer = (
            None
            if self.active_output_mode == "multiout"
            else nn.Embedding(len(self.vocab.readout_vocab), self.embedding_dim)
        )
        self.log_dose_layer = nn.Linear(1, self.embedding_dim)
        self.time_layer = nn.Linear(1, self.embedding_dim)
        self.predictor = self.build_predictor()
        if self.active_output_mode == "multiout":
            self._initialize_dataset_output_heads(len(self.vocab.dataset_vocab), len(self.vocab.readout_vocab))

    def _check_layers_initialized(self) -> None:
        if (
            self.context_embedding_layer is None
            or self.perturb_embedding_layer is None
            or self.log_dose_layer is None
            or self.time_layer is None
        ):
            raise ValueError("Embedding layers not initialized.")
        if self.active_output_mode == "multiout":
            if self.dataset_output_weight is None or self.dataset_output_bias is None:
                raise ValueError("Dataset-specific output heads not initialized.")
            return
        if self.dataset_embedding_layer is None:
            raise ValueError("Dataset embedding layer not initialized.")
        if self.active_output_mode == "scalar" and self.readout_embedding_layer is None:
            raise ValueError("Readout embedding layer not initialized.")

    def embed(self, batch: pl.DataFrame) -> tuple[torch.Tensor, ...]:  # noqa: D102
        # This method is not used during training, but we keep it for convenience.
        self._check_layers_initialized()

        if self._has_multiout_columns(batch.columns):
            encode_first = False
        else:
            encode_first = not (
                batch["context"].dtype.is_integer()
                and batch["perturbation"].dtype.is_nested()
                and batch["readout"].dtype.is_integer()
            )
        if encode_first:
            if self.vocab is None:
                raise ValueError("Vocabulary must be set before embedding.")
            enc_batch = encode_data(batch, self.vocab)
        else:
            enc_batch = batch

        tensor_dict = to_tensor_dict(enc_batch)
        # Move tensors to the same device as the embedding layers.
        assert self.context_embedding_layer is not None  # for type checkers
        device = self.context_embedding_layer.weight.device
        tensor_dict = {k: v.to(device, non_blocking=True) for k, v in tensor_dict.items()}
        return self.embed_tensor_dict(tensor_dict)

    def embed_tensor_dict(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:  # noqa: D102
        self._check_layers_initialized()
        # for type checkers — _check_layers_initialized guarantees these are not None
        assert self.context_embedding_layer is not None
        assert self.perturb_embedding_layer is not None
        assert self.log_dose_layer is not None
        assert self.time_layer is not None

        embedded_context = self.context_embedding_layer(batch["context"])
        embedded_perturb = self.perturb_embedding_layer(batch["perturbation_flat"], batch["perturbation_offset"])

        # Continuous features: cast to the same float dtype as the embedding tables (Linear
        # layers default to float32; parquet floats are float64) and add a feature dim.
        target_dtype = self.log_dose_layer.weight.dtype
        log_dose = batch["log_dose"].to(dtype=target_dtype).unsqueeze(-1)
        time = batch["time"].to(dtype=target_dtype).unsqueeze(-1)
        embedded_log_dose = self.log_dose_layer(log_dose)
        embedded_time = self.time_layer(time)

        if self.active_output_mode == "multiout":
            return (
                embedded_context,
                embedded_perturb,
                embedded_log_dose,
                embedded_time,
            )

        assert self.dataset_embedding_layer is not None
        assert self.readout_embedding_layer is not None
        embedded_dataset = self.dataset_embedding_layer(batch["dataset"])
        embedded_readout = self.readout_embedding_layer(batch["readout"])
        return (
            embedded_dataset,
            embedded_context,
            embedded_perturb,
            embedded_readout,
            embedded_log_dose,
            embedded_time,
        )

    def fit(  # noqa: D102
        self,
        traindata: PlibData[pl.DataFrame],
        valdata: PlibData[pl.DataFrame] | None = None,
        resume_from_checkpoint: str | None = None,
    ):
        self.train()

        # Honour checkpoint path passed directly to fit(); fall back to the one
        # stored at construction time (e.g. when called via training.py).
        ckpt_path = resume_from_checkpoint or self.resume_from_checkpoint
        if ckpt_path is not None:
            if not self.lightning_trainer_pars.get("enable_checkpointing", True):
                logger.warning(
                    "resume_from_checkpoint is set but enable_checkpointing is False in "
                    "lightning_trainer_pars. New checkpoints will NOT be written, so the run "
                    "cannot be resumed again if interrupted. Set enable_checkpointing: true "
                    "in your config to persist checkpoints."
                )
            logger.info(f"Resuming training from checkpoint: {ckpt_path}")

        self._initialize_vocabularies_and_embeddings(traindata)
        assert self.vocab is not None

        if self.active_output_mode == "multiout":
            traindata_tensors = to_tensor_dict(traindata)
        else:
            traindata_tensors = to_tensor_dict(encode_data(traindata, self.vocab))
        train_loader = traindata_tensors.get_data_loader(
            self.batch_size, self.num_workers, self.pin_memory, shuffle=True
        )
        if valdata is not None:
            if self.active_output_mode == "multiout":
                valdata_tensors = to_tensor_dict(valdata)
                val_batch_size = self.batch_size
            else:
                valdata_tensors = to_tensor_dict(encode_data(valdata, self.vocab))
                val_batch_size = None
            val_loader = valdata_tensors.get_data_loader(
                val_batch_size, self.num_workers, self.pin_memory, shuffle=False
            )
        else:
            val_loader = None

        trainer = pyl.Trainer(**self.lightning_trainer_pars)
        logger.info(f"Fitting {self.__class__.__name__}..")
        trainer.fit(model=self, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
        if len(self.throughput_outputs) > 1:
            avg_thr = sum(self.throughput_outputs[1:]) / len(self.throughput_outputs[1:])
            logger.info(f"Average throughput: {int(avg_thr)} samples/sec == {int(avg_thr / self.batch_size)} it/sec")
        logger.info("Model fitting completed")

    def configure_callbacks(self):  # noqa: D102
        cblist: list[Callback] = []
        if self.early_stopping_patience > 0:
            # add early stopping callback
            cblist.append(EarlyStopping("Validation RMSE", patience=self.early_stopping_patience, verbose=True))
            # add model checkpointing callback
            logger.info(f"Temporary file for checkpoints is {self.ckpt_filename}.ckpt")
            cblist.append(ModelCheckpoint(self.model_checkpoints_path, self.ckpt_filename, monitor="Validation RMSE"))
        if self.epoch_checkpoint_every_n > 0 or self.epoch_checkpoint_save_last:
            # Stock Lightning ModelCheckpoint is properly DDP-aware (every rank calls
            # trainer.save_checkpoint together). every_n_epochs >= 1 controls the cadence;
            # save_top_k=-1 keeps all snapshots; save_last writes a rolling last.ckpt at
            # the same cadence (use every_n=1 if you need per-epoch resume granularity).
            every_n = max(self.epoch_checkpoint_every_n, 1)
            logger.info(
                f"Epoch checkpoints every {every_n} epochs, "
                f"save_last={self.epoch_checkpoint_save_last} at {self.model_checkpoints_path}"
            )
            cblist.append(
                ModelCheckpoint(
                    dirpath=self.model_checkpoints_path,
                    filename="epoch-{epoch:04d}",
                    every_n_epochs=every_n,
                    save_top_k=-1,
                    save_last=self.epoch_checkpoint_save_last,
                )
            )
        return cblist

    def add_logger(self, output_dir: Path, log_name: str):  # noqa: D102
        self.lightning_trainer_pars["logger"] = TensorBoardLogger(save_dir=output_dir, name=log_name)
        # Co-locate step/epoch checkpoints with the per-run TensorBoard logs so that
        # everything for one training run lives under the same results dir (and is
        # wiped together when run.sh clears RESULTS_DIR before the next run).
        # Without this, checkpoints would land in the global .plib_cache/checkpoints
        # and accumulate across runs.
        self.model_checkpoints_path = Path(output_dir) / "checkpoints"

    @torch.no_grad()
    def predict(self, data_x: PlibData[pl.DataFrame], batch_size: int | None = 100_000) -> NDArray:  # noqa: D102
        if self.vocab is None:
            raise ValueError("Model not fitted yet.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()

        if self.active_output_mode == "multiout" and batch_size is not None:
            batch_size = min(batch_size, self.batch_size)
        batch_size = min(len(data_x), batch_size) if batch_size is not None else None
        if self._is_multiout_data(data_x):
            data_x_tensors = to_tensor_dict(data_x)
        else:
            data_x_tensors = to_tensor_dict(encode_data(data_x, self.vocab))
        data_loader = data_x_tensors.get_data_loader(batch_size, self.num_workers, self.pin_memory, shuffle=False)

        predictions_list: list[torch.Tensor] = []
        for batch in data_loader:
            batch_device = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            pred = self(batch_device)
            if self.active_output_mode == "multiout":
                predictions_list.append(self._select_predictions_for_available_readouts(pred, batch_device))
            else:
                predictions_list.append(pred)

        return torch.cat(predictions_list).cpu().detach().numpy().flatten()

    @staticmethod
    def _init_weights(module):  # noqa: D102
        if isinstance(module, (nn.Linear, nn.Embedding, nn.EmbeddingBag)):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("relu"))

    def build_predictor(self) -> nn.Module:
        """Where neural network architecture is instantiated."""
        # Scalar mode consumes 6 D-dim feature vectors including dataset/readout.
        # Multi-output mode consumes sample features only; dataset selects a
        # separate output head instead of being an input embedding.
        n_input_embeddings = 4 if self.active_output_mode == "multiout" else 6
        input_dim = n_input_embeddings * self.embedding_dim
        neural_network = nn.Sequential()
        for i in range(self.num_layers):
            neural_network.append(nn.Linear(self.hidden_dim if i > 0 else input_dim, self.hidden_dim))
            neural_network.append(nn.ReLU())
            neural_network.append(nn.Dropout(self.dropout))
        if self.active_output_mode != "multiout":
            neural_network.append(nn.Linear(self.hidden_dim if self.num_layers > 0 else input_dim, self.output_dim))
        neural_network.apply(self._init_weights)
        return neural_network

    def _predictor_output_dim(self) -> int:
        if self.num_layers > 0:
            return self.hidden_dim
        return (4 if self.active_output_mode == "multiout" else 6) * self.embedding_dim

    def _initialize_dataset_output_heads(self, num_datasets: int, num_readouts: int) -> None:
        head_input_dim = self._predictor_output_dim()
        weight = torch.empty(num_datasets, num_readouts, head_input_dim)
        bias = torch.empty(num_datasets, num_readouts)
        for dataset_idx in range(num_datasets):
            nn.init.xavier_uniform_(weight[dataset_idx], gain=nn.init.calculate_gain("linear"))
            nn.init.zeros_(bias[dataset_idx])
        self.dataset_output_weight = nn.Parameter(weight)
        self.dataset_output_bias = nn.Parameter(bias)

    def _apply_dataset_output_heads(self, features: torch.Tensor, dataset_codes: torch.Tensor) -> torch.Tensor:
        if self.dataset_output_weight is None or self.dataset_output_bias is None:
            raise ValueError("Dataset-specific output heads not initialized.")
        dataset_codes = dataset_codes.long()
        pred = features.new_empty((features.shape[0], self.output_dim))
        for dataset_code in dataset_codes.unique(sorted=True):
            mask = dataset_codes == dataset_code
            head_idx = dataset_code.item()
            pred[mask] = features[mask] @ self.dataset_output_weight[head_idx].t() + self.dataset_output_bias[head_idx]
        return pred

    @staticmethod
    def _value_row_indices(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.repeat_interleave(
            torch.arange(batch["context"].shape[0], device=batch["context"].device),
            batch["n_values"].to(device=batch["context"].device).long(),
        )

    def _select_predictions_for_available_readouts(
        self, pred: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if "readout_flat" in batch and "n_values" in batch:
            row_indices = self._value_row_indices(batch)
            return pred[row_indices, batch["readout_flat"].long()]
        if "readout" in batch:
            row_indices = torch.arange(batch["readout"].shape[0], device=batch["readout"].device)
            return pred[row_indices, batch["readout"].long()]
        raise ValueError("Multi-output prediction requires either readout_flat/n_values or readout tensors.")

    def _loss_for_batch(self, pred: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.active_output_mode == "multiout":
            selected_pred = self._select_predictions_for_available_readouts(pred, batch)
            return self.loss(selected_pred, batch["value"].to(dtype=selected_pred.dtype)).reshape(-1)
        return self.loss(pred, batch["value"].unsqueeze(-1)).squeeze().reshape(-1)

    def _context_codes_for_losses(self, batch: dict[str, torch.Tensor], unreduced_loss: torch.Tensor) -> torch.Tensor:
        if self.active_output_mode == "multiout":
            return torch.repeat_interleave(
                batch["context"], batch["n_values"].to(device=batch["context"].device).long()
            )
        return batch["context"]

    @staticmethod
    def _new_metric_buffers(num_slots: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(num_slots, device=device, dtype=torch.float64),
            torch.zeros(num_slots, device=device, dtype=torch.float64),
        )

    @staticmethod
    def _all_reduce_metric_buffers(sum_sq: torch.Tensor, count: torch.Tensor) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sum_sq, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)

    # -----------------------------------------------------
    # PyTorch Lightning-related methods
    # -----------------------------------------------------

    def configure_optimizers(self):  # noqa: D102
        optimizer_dict = {
            "Adam": torch.optim.Adam(self.parameters(), lr=self.learning_rate),
            "AdamW": torch.optim.AdamW(self.parameters(), lr=self.learning_rate),
            "Adagrad": torch.optim.Adagrad(self.parameters(), lr=self.learning_rate),
            "Adadelta": torch.optim.Adadelta(self.parameters(), lr=self.learning_rate),
            "Adamax": torch.optim.Adamax(self.parameters(), lr=self.learning_rate),
            "SGD": torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9),
            "RMSprop": torch.optim.RMSprop(self.parameters(), lr=self.learning_rate),
        }
        if self.optimizer_name not in optimizer_dict.keys():
            ValueError(f"Unrecognized optimizer {self.optimizer_name}")
        optimizer = optimizer_dict[self.optimizer_name]
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.learning_rate_decay)
        return [optimizer], [scheduler]

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:  # noqa: D102
        features = self.predictor(torch.cat(self.embed_tensor_dict(batch), dim=1))
        if self.active_output_mode == "multiout":
            return self._apply_dataset_output_heads(features, batch["dataset"])
        return features

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):  # noqa: D102
        pred: torch.Tensor = self(batch)
        unreduced_loss = self._loss_for_batch(pred, batch)
        loss_for_metric = unreduced_loss.detach()
        if self.training_loss_sum is None or self.training_loss_count is None:
            self.training_loss_sum, self.training_loss_count = self._new_metric_buffers(1, loss_for_metric.device)
        self.training_loss_sum[0] += loss_for_metric.sum(dtype=torch.float64)
        self.training_loss_count[0] += loss_for_metric.numel()
        return unreduced_loss.mean()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):  # noqa: D102
        pred = self(batch)
        unreduced_loss = self._loss_for_batch(pred, batch)

        assert self.vocab is not None
        loss_for_metric = unreduced_loss.detach()
        num_contexts = len(self.vocab.context_vocab)
        if self.validation_loss_sum is None or self.validation_loss_count is None:
            self.validation_loss_sum, self.validation_loss_count = self._new_metric_buffers(
                num_contexts + 1, loss_for_metric.device
            )

        # Slot 0 = global metric; slots 1..N are indexed by context code.
        self.validation_loss_sum[0] += loss_for_metric.sum(dtype=torch.float64)
        self.validation_loss_count[0] += loss_for_metric.numel()
        loss_context = self._context_codes_for_losses(batch, unreduced_loss).long()
        slot_indices = loss_context + 1
        self.validation_loss_sum.index_add_(0, slot_indices, loss_for_metric.to(dtype=torch.float64))
        self.validation_loss_count.index_add_(0, slot_indices, torch.ones_like(loss_for_metric, dtype=torch.float64))

    def on_train_epoch_end(self):  # noqa: D102
        if self.training_loss_sum is not None and self.training_loss_count is not None:
            self._all_reduce_metric_buffers(self.training_loss_sum, self.training_loss_count)
            metric_result = (self.training_loss_sum / self.training_loss_count.clamp(min=1)).sqrt()[0]
            self.log("Training RMSE", metric_result)
        self.training_loss_sum = None
        self.training_loss_count = None
        # Throughput is read from the tqdm progress bar, which Lightning only
        # advances on rank 0. On other ranks `format_dict["elapsed"]` is 0 and
        # the division blows up. Skip it everywhere except rank 0 (and still
        # guard against a zero elapsed for very short epochs).
        if self.global_rank != 0:
            return
        progress_bar_callback = self.trainer.progress_bar_callback
        if isinstance(progress_bar_callback, TQDMProgressBar):
            train_pbar = progress_bar_callback.train_progress_bar
            if isinstance(train_pbar, tqdm):
                samples_processed_in_epoch = train_pbar.format_dict["n"]
                time_to_process_epoch = train_pbar.format_dict["elapsed"]
                if time_to_process_epoch > 0:
                    self.throughput_outputs.append(
                        (self.batch_size * samples_processed_in_epoch) / time_to_process_epoch
                    )

    def on_validation_epoch_end(self):  # noqa: D102
        # Validation is sharded across DDP ranks (one whole shard per global worker),
        # so each rank only sees a disjoint, unequal subset of contexts. Reducing
        # locally would (a) bias the TB metric to rank 0's slice and (b) make per-rank
        # `self.log` calls asymmetric in keys/order, which is a known DDP-desync trap
        # at the epoch boundary. Instead we aggregate fixed-shape (sum_sq, count)
        # tensors over all ranks via two tiny all_reduces, then log the global RMSE.
        assert self.vocab is not None
        num_contexts = len(self.vocab.context_vocab)
        if self.validation_loss_sum is None or self.validation_loss_count is None:
            sum_sq, count = self._new_metric_buffers(num_contexts + 1, self.device)
        else:
            sum_sq = self.validation_loss_sum
            count = self.validation_loss_count

        self._all_reduce_metric_buffers(sum_sq, count)

        rmse = (sum_sq / count.clamp(min=1)).sqrt()
        # Single GPU->CPU sync for the per-slot count, used to skip empty contexts.
        count_cpu = count.detach().cpu()

        # After the all_reduce, every rank holds identical sum_sq/count, so these
        # self.log calls are now symmetric across ranks (same keys, same order).
        self.log("Validation RMSE", rmse[0])
        for symbol, code in zip(self.vocab.context_vocab["symbol"], self.vocab.context_vocab["code"]):
            if count_cpu[code + 1].item() > 0:
                self.log(f"Validation RMSE {symbol}", rmse[code + 1])

        self.validation_loss_sum = None
        self.validation_loss_count = None

    def on_fit_end(self):  # noqa: D102
        # at the end of training
        logger.info("Cleaning up...")
        # if applicable, load the best model, the one that minimizes validation loss
        if self.ckpt_filename is not None:
            path_to_checkpoint = self.model_checkpoints_path / (self.ckpt_filename + ".ckpt")
            best_model = LPM.load_from_checkpoint(path_to_checkpoint, **self.hparams)
            self.load_state_dict(best_model.state_dict())
            path_to_checkpoint.unlink()
        # detach the trainer from the model
        self.lightning_trainer_pars = {}
        self.trainer = None  # type: ignore

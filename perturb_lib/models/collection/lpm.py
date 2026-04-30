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
from collections import defaultdict
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


class CustomEpochCheckpoint(Callback):
    """Save full Lightning checkpoints on a custom epoch schedule.

    Saves at every epoch in ``special_epochs`` and additionally at every multiple of
    ``every_n`` (when ``every_n > 0``). Files are named ``epoch-NNNN.ckpt``. Saves only on
    the global rank 0 to avoid contention on shared filesystems; all other ranks no-op.

    No rolling deletion: every checkpoint that the schedule produces is kept on disk
    forever (until removed manually). Disk usage is the caller's responsibility.

    The format written by ``trainer.save_checkpoint`` is the standard Lightning checkpoint
    (model + optimizer + LR scheduler + global_step + current_epoch + RNG), so any file
    written here is drop-in compatible with ``resume_from_checkpoint`` / ``ckpt_path``.
    """

    def __init__(
        self,
        dirpath: Path,
        special_epochs: tuple[int, ...] = (),
        every_n: int = 0,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.special_epochs = set(special_epochs)
        self.every_n = every_n

    def _should_save(self, epoch_one_indexed: int) -> bool:
        if epoch_one_indexed in self.special_epochs:
            return True
        if self.every_n > 0 and epoch_one_indexed >= self.every_n and epoch_one_indexed % self.every_n == 0:
            return True
        return False

    def on_train_epoch_end(self, trainer, pl_module):  # noqa: D102
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1  # current_epoch is 0-indexed; humans count from 1
        if not self._should_save(epoch):
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.dirpath / f"epoch-{epoch:04d}.ckpt"
        trainer.save_checkpoint(str(ckpt_path))


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
        epoch_checkpoint_special: Specific epoch numbers (1-indexed) at which to save a full
            Lightning checkpoint. Combined (union) with ``epoch_checkpoint_every_n``.
        epoch_checkpoint_every_n: Save a checkpoint at every multiple of this number of
            epochs (e.g. 5 -> epochs 5, 10, 15, ...). Set to 0 to disable; only the
            ``special`` epochs are then saved. No rolling deletion is done -- callers are
            responsible for cleaning up old checkpoints if disk usage matters.
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
        epoch_checkpoint_special: list[int] | tuple[int, ...] = (),
        epoch_checkpoint_every_n: int = 0,
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
        # Epoch-based checkpoint schedule. We save at every epoch in `epoch_checkpoint_special`
        # (e.g. [1] for an epoch-1 sanity snapshot) and additionally at every multiple of
        # `epoch_checkpoint_every_n` (e.g. 5 -> epochs 5, 10, 15, ...). Set every_n=0 to only save
        # the special epochs; set special=() to only save every Nth. No rolling deletion: all
        # saved checkpoints are kept on disk until manually removed.
        self.epoch_checkpoint_special = tuple(epoch_checkpoint_special)
        self.epoch_checkpoint_every_n = epoch_checkpoint_every_n
        self.optimizer_name = optimizer_name
        self.embedding_aggregation_mode = embedding_aggregation_mode
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
        # Continuous-feature projections: a single scalar -> embedding_dim vector.
        self.log_dose_layer: nn.Linear | None = None
        self.time_layer: nn.Linear | None = None

        # prediction neural network
        self.predictor = self.build_predictor()

        # PL-related
        self.training_step_outputs: list[torch.Tensor] = []
        self.validation_step_outputs: list[torch.Tensor] = []
        self.throughput_outputs: list[float] = []
        self.validation_step_per_context_outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
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
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):  # noqa: D102
        state_dict = cast(dict[str, Any], state_dict)  # to avoid MyPy complaining about "pop" operation
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
        if "context_embedding_layer.weight" in state_dict:
            self.context_embedding_layer = nn.Embedding.from_pretrained(state_dict["context_embedding_layer.weight"])
        if "perturb_embedding_layer.weight" in state_dict:
            self.perturb_embedding_layer = nn.EmbeddingBag.from_pretrained(state_dict["perturb_embedding_layer.weight"])
        if "readout_embedding_layer.weight" in state_dict:
            self.readout_embedding_layer = nn.Embedding.from_pretrained(state_dict["readout_embedding_layer.weight"])
        if "dataset_embedding_layer.weight" in state_dict:
            self.dataset_embedding_layer = nn.Embedding.from_pretrained(state_dict["dataset_embedding_layer.weight"])
        super().load_state_dict(state_dict, strict, assign)

    def _initialize_vocabularies_and_embeddings(self, data: PlibData):
        self.vocab = Vocabulary.initialize_from_data(data)

        self.dataset_embedding_layer = nn.Embedding(len(self.vocab.dataset_vocab), self.embedding_dim)
        self.context_embedding_layer = nn.Embedding(len(self.vocab.context_vocab), self.embedding_dim)
        self.perturb_embedding_layer = nn.EmbeddingBag(
            len(self.vocab.perturb_vocab), self.embedding_dim, mode=self.embedding_aggregation_mode
        )
        self.readout_embedding_layer = nn.Embedding(len(self.vocab.readout_vocab), self.embedding_dim)
        self.log_dose_layer = nn.Linear(1, self.embedding_dim)
        self.time_layer = nn.Linear(1, self.embedding_dim)

    def _check_layers_initialized(self) -> None:
        if (
            self.dataset_embedding_layer is None
            or self.context_embedding_layer is None
            or self.perturb_embedding_layer is None
            or self.readout_embedding_layer is None
            or self.log_dose_layer is None
            or self.time_layer is None
        ):
            raise ValueError("Embedding layers not initialized.")

    def embed(  # noqa: D102
        self, batch: pl.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # This method is not used during training, but we keep it for convenience.
        self._check_layers_initialized()

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
        assert self.readout_embedding_layer is not None  # for type checkers
        device = self.readout_embedding_layer.weight.device
        tensor_dict = {k: v.to(device, non_blocking=True) for k, v in tensor_dict.items()}
        return self.embed_tensor_dict(tensor_dict)

    def embed_tensor_dict(  # noqa: D102
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._check_layers_initialized()
        # for type checkers — _check_layers_initialized guarantees these are not None
        assert self.dataset_embedding_layer is not None
        assert self.context_embedding_layer is not None
        assert self.perturb_embedding_layer is not None
        assert self.readout_embedding_layer is not None
        assert self.log_dose_layer is not None
        assert self.time_layer is not None

        embedded_dataset = self.dataset_embedding_layer(batch["dataset"])
        embedded_context = self.context_embedding_layer(batch["context"])
        embedded_perturb = self.perturb_embedding_layer(
            batch["perturbation_flat"], batch["perturbation_offset"]
        )
        embedded_readout = self.readout_embedding_layer(batch["readout"])

        # Continuous features: cast to the same float dtype as the embedding tables (Linear
        # layers default to float32; parquet floats are float64) and add a feature dim.
        target_dtype = self.log_dose_layer.weight.dtype
        log_dose = batch["log_dose"].to(dtype=target_dtype).unsqueeze(-1)
        time = batch["time"].to(dtype=target_dtype).unsqueeze(-1)
        embedded_log_dose = self.log_dose_layer(log_dose)
        embedded_time = self.time_layer(time)

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

        traindata_tensors = to_tensor_dict(encode_data(traindata, self.vocab))
        train_loader = traindata_tensors.get_data_loader(
            self.batch_size, self.num_workers, self.pin_memory, shuffle=True
        )
        if valdata is not None:
            valdata_tensors = to_tensor_dict(encode_data(valdata, self.vocab))
            val_loader = valdata_tensors.get_data_loader(None, self.num_workers, self.pin_memory, shuffle=False)
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
        if self.epoch_checkpoint_special or self.epoch_checkpoint_every_n > 0:
            # Custom epoch-based checkpointing: writes at the union of `special_epochs`
            # and every Nth epoch. No rolling deletion -- all saved checkpoints are kept.
            logger.info(
                f"Epoch checkpoints at special={sorted(self.epoch_checkpoint_special)} "
                f"and every {self.epoch_checkpoint_every_n} epochs at {self.model_checkpoints_path}"
            )
            cblist.append(
                CustomEpochCheckpoint(
                    dirpath=self.model_checkpoints_path,
                    special_epochs=tuple(self.epoch_checkpoint_special),
                    every_n=self.epoch_checkpoint_every_n,
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

        batch_size = min(len(data_x), batch_size) if batch_size is not None else None
        data_x_tensors = to_tensor_dict(encode_data(data_x, self.vocab))
        data_loader = data_x_tensors.get_data_loader(batch_size, self.num_workers, self.pin_memory, shuffle=False)

        predictions_list: list[torch.Tensor] = []
        for batch in data_loader:
            batch_device = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            predictions_list.append(self(batch_device))

        return torch.cat(predictions_list).cpu().detach().numpy().flatten()

    @staticmethod
    def _init_weights(module):  # noqa: D102
        if isinstance(module, (nn.Linear, nn.Embedding, nn.EmbeddingBag)):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("relu"))

    def build_predictor(self) -> nn.Module:
        """Where neural network architecture is instantiated."""
        # 6 D-dim feature vectors: dataset, context, perturbation, readout, log_dose, time.
        input_dim = 6 * self.embedding_dim
        neural_network = nn.Sequential()
        for i in range(self.num_layers):
            neural_network.append(nn.Linear(self.hidden_dim if i > 0 else input_dim, self.hidden_dim))
            neural_network.append(nn.ReLU())
            neural_network.append(nn.Dropout(self.dropout))
        neural_network.append(nn.Linear(self.hidden_dim if self.num_layers > 0 else input_dim, 1))
        neural_network.apply(self._init_weights)
        return neural_network

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
        return self.predictor(torch.cat(self.embed_tensor_dict(batch), dim=1))

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):  # noqa: D102
        pred: torch.Tensor = self(batch)
        unreduced_loss: torch.Tensor = self.loss(pred, batch["value"].unsqueeze(-1)).squeeze()
        self.training_step_outputs.append(unreduced_loss.detach())  # detach() to not keep the graph alive
        return unreduced_loss.mean()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):  # noqa: D102
        pred = self(batch)
        unreduced_loss = self.loss(pred, batch["value"].unsqueeze(-1)).squeeze()
        self.validation_step_outputs.append(unreduced_loss)

        assert self.vocab is not None
        context_codes = torch.unique(batch["context"])
        # NOTE: this will cause a GPU-CPU synchronization, but can't be avoided since the vocabulary is on the CPU
        context_codes_series = pl.from_numpy(context_codes.cpu().numpy(), schema=["context"])["context"]
        context_symbol_series = context_codes_series.replace_strict(
            self.vocab.context_vocab["code"], self.vocab.context_vocab["symbol"]
        )

        for context_code, context_symbol in zip(context_codes_series, context_symbol_series):
            context_mask = batch["context"] == context_code
            unreduced_context_loss = unreduced_loss[context_mask]
            self.validation_step_per_context_outputs[context_symbol].append(unreduced_context_loss)

    @staticmethod
    def _reduce_outputs(outputs):
        return "RMSE", torch.cat(outputs).mean().sqrt()

    def on_train_epoch_end(self):  # noqa: D102
        metric_label, metric_result = self._reduce_outputs(self.training_step_outputs)
        self.log(f"Training {metric_label}", metric_result)
        self.training_step_outputs.clear()
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
        metric_label, metric_result = self._reduce_outputs(self.validation_step_outputs)
        self.log(f"Validation {metric_label}", metric_result)
        for context in self.validation_step_per_context_outputs.keys():
            metric_label, metric_result = self._reduce_outputs(self.validation_step_per_context_outputs[context])
            self.log(f"Validation {metric_label} {context}", metric_result)
        self.validation_step_outputs.clear()
        self.validation_step_per_context_outputs.clear()

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

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

Data-related utility functions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Generic, Literal, Protocol, cast, runtime_checkable

import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import torch.utils.data
from typing_extensions import TypeVar

from perturb_lib._utils import select_random_subset
from perturb_lib.data import ControlSymbol
from perturb_lib.environment import get_seed, logger
from perturb_lib.utils import get_rank_info

VALDATA_PORTION = 0.15
TESTDATA_PORTION = 0.15
SEED = 13

# We have to use the older TypeVar syntax because the new syntax causes issues when pickling during multiprocessing
TCInT = TypeVar("TCInT")  # TransformChain input type
TCOutT = TypeVar("TCOutT")  # TransformChain output type
TCNewOutT = TypeVar("TCNewOutT")  # TransformChain new output type
SBOutT = TypeVar("SBOutT", default=pl.DataFrame)  # ShuffleBuffer output type


ModelSystemType = Literal[
    "HumanCellLine",
    "hiPSC-derivedGlutamatergicNeurons",
    "hiPSC-derivedAstrocytes",
    "PBMC-derivedBCells",
    "PBMC-derivedTCells",
    "PBMC-derivedNKCells",
    "PBMC-derivedMyeloidCells",
]

TechnologyType = Literal[
    "10xChromium3-scRNA-seq",
    "10xChromium5-scRNA-seq",
    "Mosaic-scRNA-seq",
    "GrowthScreen",
    "L1000-RNA-seq",
]


def anndata_format_verification(adata: sc.AnnData) -> None:
    """Basic AnnData format verification."""
    if "readout" not in adata.var.columns:
        raise ValueError("Badly formatted AnnData: 'readout' is missing in 'var'.")
    if "perturbation" not in adata.obs.columns:
        raise ValueError("Badly formatted AnnData: 'perturbation' is missing in 'obs'.")


def _get_random_split(perturbations: Iterable[str]) -> tuple[set[str], set[str], set[str]]:
    control_exists = ControlSymbol in set(perturbations)
    all_perturbations_excluding_control = sorted(set(perturbations).difference({ControlSymbol}))
    num_of_val_perturbations = int(len(all_perturbations_excluding_control) * VALDATA_PORTION)
    num_of_test_perturbations = int(len(all_perturbations_excluding_control) * TESTDATA_PORTION)
    val_perturbations = select_random_subset(all_perturbations_excluding_control, num_of_val_perturbations, seed=SEED)
    remaining_perturbations = sorted(set(all_perturbations_excluding_control) - set(val_perturbations))
    test_perturbations = select_random_subset(remaining_perturbations, num_of_test_perturbations, seed=SEED)
    train_perturbations = set(remaining_perturbations).difference(test_perturbations)
    if control_exists:
        train_perturbations.add(ControlSymbol)
    return train_perturbations, val_perturbations, test_perturbations


def add_train_test_val_splits(adata: sc.AnnData):
    """Adding train/test/validation splits to the given AnnData object."""
    logger.info("Adding train/val/test splits..")
    trainset, valset, testset = _get_random_split(adata.obs.perturbation)
    split_vector = pd.Series([""] * len(adata.obs), index=adata.obs.perturbation)
    split_vector.loc[list(trainset)] = "train"
    split_vector.loc[list(valset)] = "val"
    split_vector.loc[list(testset)] = "test"
    assert "" not in set(split_vector), "Some samples have not been assigned to train/val/test."
    adata.obs["split"] = split_vector.values


class TransformChain(Generic[TCInT, TCOutT]):
    """Chain of functions that can be applied to a single input.

    This class support static type-checking. For example:

    def int_to_str(x: int) -> str:
        return str(x)

    chain: TransformChain[int, int] = TransformChain()  # type of 'chain' is TransformChain[int, int]
    chain2 = chain.append(int_to_str)  # type of 'chain2' correctly inferred as TransformChain[int, str]
    """

    def __init__(self):
        self.callables: list[Callable] = []

    def append(self, function: Callable[[TCOutT], TCNewOutT]) -> TransformChain[TCInT, TCNewOutT]:
        """Append a function to the chain."""
        chain = TransformChain[TCInT, TCNewOutT]()
        chain.callables = self.callables + [function]
        return chain

    def __call__(self, input_: TCInT) -> TCOutT:
        """Apply the chain of functions to the input."""
        for function in self.callables:
            input_ = function(input_)

        # Since we don't keep track of the precise types of the functions in the chain, we need to cast the output
        return cast(TCOutT, input_)


@runtime_checkable
class ShardedIterableDataset(Protocol):
    """Protocol for sharded iterable datasets."""

    def _iterate_shards(
        self,
        seed: int | None,
        start_shard_idx: int | None = None,
        end_shard_idx: int | None = None,
        shard_step_size: int = 1,
        shard_indices: Sequence[int] | None = None,
    ) -> Iterator[pl.DataFrame]:
        """Iterate over shards of data.

        Args:
            seed: Shuffle shard if a seed is provided. Otherwise, do not shuffle.
            start_shard_idx: Start iterating from this shard index.
            end_shard_idx: Stop iterating before reaching this shard index.
            shard_step_size: Step size for iterating over shards
            shard_indices: Explicit shard indices to iterate. If provided, start/stop/step are ignored.

        Returns: Iterator over shards of data.

        """

    def _num_shards(self) -> int:
        """Get the number of shards in the dataset."""

    def __len__(self) -> int:
        """Get the number of samples in the dataset"""


class ShuffleBuffer(torch.utils.data.IterableDataset[SBOutT], Generic[SBOutT]):
    """Buffer chunks of data and yield them as batches while optionally shuffling the buffer.

    Args:
        dataset: ShardedIterableDataset to sample from.
        batch_size: Size of the batches to yield. If batch_size = None, no batching is done and the full shard is returned
        refill_threshold: Maximum number of samples to keep in the buffer before consolidating and shuffling.
            If batch_size is None, this parameter is ignored.
        shuffle: Whether to shuffle the buffer.
        transforms: Chain of transformations to apply to each batch.
    """

    def __init__(
        self,
        dataset: ShardedIterableDataset,
        batch_size: int | None,
        refill_threshold: int = 1_000_000,
        shuffle: bool = True,
        transforms: TransformChain[pl.DataFrame, SBOutT] | None = None,
    ):
        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError("Batch size must be positive.")

            if shuffle is False:
                refill_threshold = batch_size

            if refill_threshold < batch_size:
                raise ValueError("Buffer refill threshold must be at least equal to batch_size.")
        else:
            # By setting refill_threshold to 0, we ensure that the buffer is immediately returned after every shard
            refill_threshold = 0

        self.dataset = dataset
        self.batch_size = batch_size
        self.refill_threshold = refill_threshold
        self.shuffle = shuffle
        self.transforms: TransformChain[pl.DataFrame, SBOutT] = (
            transforms if transforms is not None else TransformChain()
        )

        self._iter_counter = 0
        self._start_shard_idx: int | None = None
        self._stop_shard_idx: int | None = None
        self._shard_step_size = 1
        self._num_workers_for_len = 0

        self._buffer: list[pl.DataFrame] = []

    @property
    def _num_samples_in_buffer(self) -> int:
        return sum(len(shard) for shard in self._buffer)

    def _get_seed_for_current_iteration(self) -> int | None:
        if self.shuffle:
            return get_seed() + self._iter_counter

        return None

    def _add_shard(self, shard: pl.DataFrame):
        if self.shuffle and (len(shard) > self.refill_threshold):
            logger.warning(
                f"Shard of length {len(shard)} exceeds buffer refill threshold of {self.refill_threshold}. "
                "This might result in a buffer with only one shard and low shuffling efficiency."
            )
        self._buffer.append(shard)

    def _consolidate_and_shuffle_buffer(self):
        self._buffer = [pl.concat(self._buffer, how="vertical", rechunk=True)]
        if self.shuffle:
            self._buffer[0] = self._buffer[0].sample(
                fraction=1, with_replacement=False, shuffle=True, seed=self._get_seed_for_current_iteration()
            )

    def _pop_batch(self) -> SBOutT:
        # Assume that buffer is of length 1
        if self.batch_size is not None:
            batch = self._buffer[0].head(self.batch_size)
            self._buffer[0] = self._buffer[0].tail(-self.batch_size)
        else:
            # Automatic batching is disabled, therefore return the full shard
            batch = self._buffer[0]
            self._buffer = []
        return self.transforms(batch)

    def _local_shard_indices_for_len(self) -> list[int]:
        balanced_indices = self._balanced_shard_indices_by_rank()
        if balanced_indices is not None:
            return balanced_indices[get_rank_info().rank]

        rank_info = get_rank_info()
        num_shards = self.dataset._num_shards()
        if num_shards == 0:
            return []

        if self._start_shard_idx is not None or self._stop_shard_idx is not None or self._shard_step_size != 1:
            start = 0 if self._start_shard_idx is None else self._start_shard_idx
            stop = num_shards if self._stop_shard_idx is None else self._stop_shard_idx
            return list(range(start, stop, self._shard_step_size))

        num_workers = max(self._num_workers_for_len, 0)
        if num_workers > 0:
            global_num_workers = rank_info.world_size * num_workers
            starts = range(rank_info.rank * num_workers, (rank_info.rank + 1) * num_workers)
            return [idx for start in starts for idx in range(start, num_shards, global_num_workers)]

        return list(range(rank_info.rank, num_shards, rank_info.world_size))

    def _shard_sizes(self) -> list[int] | None:
        data = getattr(self.dataset, "_data", None)
        if isinstance(data, pl.DataFrame) and "size" in data.columns:
            return [int(size) for size in data["size"].to_list()]
        return None

    def _balanced_shard_indices_by_rank(self) -> list[list[int]] | None:
        rank_info = get_rank_info()
        if rank_info.world_size <= 1:
            return None

        shard_sizes = self._shard_sizes()
        if shard_sizes is None:
            return None

        # Largest-processing-time scheduling keeps DDP ranks close in number of samples.
        # That matters for IterableDataset DDP: if one rank has extra batches, the ranks
        # can enter different collectives and deadlock.
        ranked_shards = sorted(range(len(shard_sizes)), key=lambda idx: (shard_sizes[idx], -idx), reverse=True)
        rank_indices: list[list[int]] = [[] for _ in range(rank_info.world_size)]
        rank_loads = [0 for _ in range(rank_info.world_size)]
        for shard_idx in ranked_shards:
            rank = min(range(rank_info.world_size), key=lambda idx: (rank_loads[idx], idx))
            rank_indices[rank].append(shard_idx)
            rank_loads[rank] += shard_sizes[shard_idx]

        return rank_indices

    def _assigned_shard_indices(self, seed: int | None) -> list[int] | None:
        rank_info = get_rank_info()
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = max(self._num_workers_for_len, 0) if worker_info is None else worker_info.num_workers

        rank_indices = self._balanced_shard_indices_by_rank()
        if rank_indices is None:
            return None

        local_indices = rank_indices[rank_info.rank]
        if num_workers > 1:
            shard_sizes = self._shard_sizes()
            assert shard_sizes is not None
            worker_indices: list[list[int]] = [[] for _ in range(num_workers)]
            worker_loads = [0 for _ in range(num_workers)]
            for shard_idx in sorted(local_indices, key=lambda idx: (shard_sizes[idx], -idx), reverse=True):
                worker = min(range(num_workers), key=lambda idx: (worker_loads[idx], idx))
                worker_indices[worker].append(shard_idx)
                worker_loads[worker] += shard_sizes[shard_idx]
            local_indices = worker_indices[worker_id]

        if seed is not None and len(local_indices) > 1:
            local_indices = list(local_indices)
            np.random.RandomState(seed).shuffle(local_indices)

        return local_indices

    def _batch_count_for_shards(self, shard_indices: list[int]) -> int:
        if self.batch_size is None:
            return len(shard_indices)

        shard_sizes = self._shard_sizes()
        if shard_sizes is None:
            return math.ceil(len(self.dataset) * len(shard_indices) / self.dataset._num_shards() / self.batch_size)

        return math.ceil(sum(shard_sizes[idx] for idx in shard_indices) / self.batch_size)

    def _balanced_batch_count_for_len(self) -> int | None:
        rank_info = get_rank_info()
        if rank_info.world_size <= 1:
            return None

        rank_indices = self._balanced_shard_indices_by_rank()
        if rank_indices is None:
            return None

        num_workers = max(self._num_workers_for_len, 0)
        shard_sizes = self._shard_sizes()
        assert shard_sizes is not None

        rank_batch_counts = []
        for local_indices in rank_indices:
            if num_workers > 1:
                worker_indices: list[list[int]] = [[] for _ in range(num_workers)]
                worker_loads = [0 for _ in range(num_workers)]
                for shard_idx in sorted(local_indices, key=lambda idx: (shard_sizes[idx], -idx), reverse=True):
                    worker = min(range(num_workers), key=lambda idx: (worker_loads[idx], idx))
                    worker_indices[worker].append(shard_idx)
                    worker_loads[worker] += shard_sizes[shard_idx]
                rank_batch_counts.append(sum(self._batch_count_for_shards(indices) for indices in worker_indices))
            else:
                rank_batch_counts.append(self._batch_count_for_shards(local_indices))

        return min(rank_batch_counts)

    def _local_sample_count_for_len(self) -> int:
        shard_indices = self._local_shard_indices_for_len()
        if not shard_indices:
            return 0
        data = getattr(self.dataset, "_data", None)
        if isinstance(data, pl.DataFrame) and "size" in data.columns:
            return int(data["size"].gather(shard_indices).sum())
        return math.ceil(len(self.dataset) * len(shard_indices) / self.dataset._num_shards())

    def __len__(self):
        balanced_batch_count = self._balanced_batch_count_for_len()
        if balanced_batch_count is not None:
            return balanced_batch_count

        if self.batch_size is None:
            # noinspection PyProtectedMember
            return len(self._local_shard_indices_for_len())
        else:
            return math.ceil(self._local_sample_count_for_len() / self.batch_size)

    def __iter__(self):
        self._iter_counter += 1
        self._buffer = []

        seed = self._get_seed_for_current_iteration()
        assigned_shard_indices = self._assigned_shard_indices(seed)

        if assigned_shard_indices is not None:
            start_shard_idx = None
            shard_step_size = 1
        elif self._start_shard_idx is None and self._stop_shard_idx is None and self._shard_step_size == 1:
            rank_info = get_rank_info()
            start_shard_idx = rank_info.rank
            shard_step_size = rank_info.world_size
        else:
            start_shard_idx = self._start_shard_idx
            shard_step_size = self._shard_step_size

        iterate_kwargs = {
            "seed": seed if assigned_shard_indices is None else None,
            "start_shard_idx": start_shard_idx,
            "end_shard_idx": self._stop_shard_idx,
            "shard_step_size": shard_step_size,
        }
        if assigned_shard_indices is not None:
            iterate_kwargs["shard_indices"] = assigned_shard_indices

        # noinspection PyProtectedMember
        for shard in self.dataset._iterate_shards(**iterate_kwargs):
            self._add_shard(shard)

            if self._num_samples_in_buffer > self.refill_threshold:
                # Once we have enough samples, concatenate all dataframes and shuffle them if needed
                self._consolidate_and_shuffle_buffer()  # After this step, buffer is a list with 1 shuffled dataframe only
                # Then pop batches until we fall below the threshold
                while self._num_samples_in_buffer > self.refill_threshold:
                    yield self._pop_batch()

        # Return the remaining samples one batch at a time
        if self._num_samples_in_buffer > 0:
            self._consolidate_and_shuffle_buffer()
            while self._num_samples_in_buffer > 0:
                yield self._pop_batch()

    @staticmethod
    def worker_init_fn(worker_id: int):
        """Worker initialization function for shuffle buffer to be used with PyTorch DataLoader.

        Ensures that each worker processes a different subset of shards.
        """
        rank_info = get_rank_info()
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None, "This function should only be called from a worker process!"

        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        global_worker_id = rank_info.rank * num_workers + worker_id
        global_num_workers = rank_info.world_size * num_workers

        buffer: ShuffleBuffer = cast(ShuffleBuffer, worker_info.dataset)

        buffer._start_shard_idx = global_worker_id
        buffer._stop_shard_idx = None  # Iterate till the end
        buffer._shard_step_size = global_num_workers  # Each worker processes a different subset of shards

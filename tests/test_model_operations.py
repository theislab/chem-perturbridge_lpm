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
"""

from typing import cast

import numpy as np
import polars as pl
import pytest

import perturb_lib as plib


def _write_multiout_plibdata(root):
    source_dir = root / "multiout_dummy"
    source_dir.mkdir(parents=True)
    rows = [
        (0, 0, [1], 0.0, 24.0, "train", [0, 1], [1.0, 2.0], 2),
        (0, 0, [1], 1.0, 24.0, "train", [1, 2], [3.0, 4.0], 2),
        (0, 0, [0], -5.0, 24.0, "val", [0, 2], [0.5, 1.5], 2),
        (0, 0, [1], 2.0, 48.0, "test", [0, 1, 2], [2.0, 1.0, 0.0], 3),
    ]
    df = pl.DataFrame(
        rows,
        schema=[
            "dataset_code",
            "context_code",
            "perturbation_codes",
            "log_dose",
            "time",
            "split",
            "readout_codes",
            "values",
            "n_values",
        ],
        orient="row",
    )
    metadata_rows = []
    for shard_idx, split in enumerate(["train", "val", "test"]):
        shard = df.filter(pl.col("split") == split)
        shard_path = f"multiout_dummy/shard_{shard_idx:06d}.parquet"
        shard.write_parquet(root / shard_path)
        metadata_rows.append(
            {
                "shard_path": shard_path,
                "source": "multiout_dummy",
                "size": shard.height,
                "scalar_values": int(shard["n_values"].sum()),
                "split": split,
                "context": "C0",
                "datasets": ["D0"],
                "contexts": ["C0"],
                "perturbations": ["Control", "P1"],
                "readouts": ["R0", "R1", "R2"],
            }
        )
    metadata = pl.DataFrame(
        metadata_rows,
        schema={
            "shard_path": pl.String,
            "source": pl.String,
            "size": pl.Int64,
            "scalar_values": pl.Int64,
            "split": pl.String,
            "context": pl.String,
            "datasets": pl.List(pl.String),
            "contexts": pl.List(pl.String),
            "perturbations": pl.List(pl.String),
            "readouts": pl.List(pl.String),
        },
    )
    metadata.write_parquet(source_dir / "metadata.parquet")
    return plib.OnDiskPlibData(data_sources=["multiout_dummy"], path_to_data_sources=root)


@pytest.mark.parametrize("model_name", ["GlobalMean", "ReadoutMean", "NoPerturb", "Catboost", "LPM"])
@pytest.mark.parametrize("plib_data", ["in_memory_plib_data", "on_disk_plib_data"])
def test_models_can_fit_predict(model_name: str, plib_data: plib.PlibData, request):
    plib_data = request.getfixturevalue(plib_data)

    model_to_args: dict[str, dict] = {
        "LPM": {
            "optimizer_name": "AdamW",
            "learning_rate": 0.001,
            "learning_rate_decay": 0.999,
            "num_layers": 1,
            "hidden_dim": 200,
            "batch_size": 10000,
            "embedding_dim": 10,
            "lightning_trainer_pars": {
                "max_epochs": 1,
                "logger": False,
                "accelerator": "cpu",
                "enable_checkpointing": False,
            },
        },
        "Catboost": {"embedding_id": "DummyEmbedding"},
    }

    model_args = model_to_args.get(model_name, {})
    model: plib.ModelMixin = plib.load_model(model_name, model_args)

    traindata, valdata, testdata = plib.split_plibdata_3fold(plib_data, "DummyData")
    testdata_x = testdata.subset_columnwise(["context", "perturbation", "readout"])

    model.fit(traindata, valdata)
    predictions = model.predict(testdata_x)
    assert isinstance(predictions, np.ndarray)


def test_model_operations():
    plib.logger.debug("Running model operations test...")
    context = "DummyData"
    data = plib.load_plibdata(context, plibdata_type=plib.InMemoryPlibData)
    traindata, valdata, _ = plib.split_plibdata_3fold(data, context)
    traindata_x = traindata.subset_columnwise(["context", "perturbation", "readout"])
    mean_value = cast(float, pl.concat([traindata[:]])["value"].mean())
    plib.logger.debug(f"Mean value across dataset: {mean_value}")

    plib.logger.debug(f"Available models: {plib.list_models()}")

    plib.logger.debug("Test GlobalMean")
    global_mean = plib.load_model("GlobalMean")
    global_mean.fit(traindata, valdata)
    predictions = global_mean.predict(traindata_x)
    plib.logger.debug(f"GlobalMean predictions: {predictions}")
    assert (predictions - mean_value < 0.0001).all(), "GlobalMean gone bad!"

    plib.logger.debug("Test pytorch model")
    mlp = plib.load_model(
        "LPM",
        model_args={
            "optimizer_name": "AdamW",
            "learning_rate": 0.001,
            "learning_rate_decay": 0.999,
            "num_layers": 1,
            "hidden_dim": 200,
            "batch_size": 10000,
            "embedding_dim": 10,
            "lightning_trainer_pars": {
                "max_epochs": 1,
                "logger": False,
                "accelerator": "cpu",
                "enable_checkpointing": False,
            },
        },
    )
    mlp.fit(traindata, valdata)
    predictions = mlp.predict(traindata_x)
    plib.logger.debug(f"Predictions: {predictions}")

    plib.logger.debug("Test successfully executed!")


def test_lpm_can_fit_predict_multiout(tmp_path):
    plib_data = _write_multiout_plibdata(tmp_path)
    traindata, valdata = plib.split_plibdata_2fold(plib_data, None)

    model = plib.load_model(
        "LPM",
        model_args={
            "optimizer_name": "AdamW",
            "learning_rate": 0.001,
            "learning_rate_decay": 0.999,
            "num_layers": 1,
            "hidden_dim": 16,
            "batch_size": 2,
            "embedding_dim": 4,
            "output_mode": "multiout",
            "lightning_trainer_pars": {
                "max_epochs": 1,
                "logger": False,
                "accelerator": "cpu",
                "enable_checkpointing": False,
                "enable_progress_bar": False,
            },
        },
    )
    model.fit(traindata, valdata)

    predict_input = valdata.subset_columnwise(
        ["dataset_code", "context_code", "perturbation_codes", "log_dose", "time", "readout_codes", "n_values"]
    )
    predictions = model.predict(predict_input)
    expected_n_predictions = int(valdata[:]["n_values"].sum())

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (expected_n_predictions,)

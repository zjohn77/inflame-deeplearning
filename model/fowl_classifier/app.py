import json
import os
from typing import Dict, Union

import mlflow
import tomli
import torch

from model.fowl_classifier import (
    load_model,
    MODULE_ROOT_DIR,
    run_inference,
    TrainImgClassifier,
)


def load_model_config(
    model_config_path: Union[os.PathLike, str, bytes]
) -> Dict[str, str]:
    """Read the model config toml; make the dirs specified within."""
    with open(model_config_path, "rb") as fp:
        _model_config = tomli.load(fp)

    # Make the output dirs if they don't already exist.
    for p in _model_config["io"].values():
        if p.split("_")[-1] == "output":
            if not os.path.isdir(p):
                os.mkdir(p)

    return _model_config


def train_model_and_serialize(_model_config: dict) -> None:
    mlflow.start_run()

    # Train model and get the best model
    train_img_classifier = TrainImgClassifier()
    best_model = train_img_classifier(
        input_data_dir=_model_config["io"]["training_input"]
    )

    # Serializing it as model.pt to the specified output directory
    output_dir = _model_config["io"]["training_output"]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    torch.save(best_model, os.path.join(output_dir, "model.pt"))

    mlflow.end_run()


def predict(_model_config: dict) -> None:
    # Loaded trained model from expected dir, and pass a sample image to it
    model = load_model(_model_config["io"]["training_output"])
    prediction: dict = run_inference(model, _model_config["io"]["inference_input"])

    # Write this prediction dict to file
    output_path = os.path.join(
        _model_config["io"]["inference_output"],
        "prediction.json",
    )
    with open(output_path, "w") as fp:
        json.dump(prediction, fp)


if __name__ == "__main__":
    model_config: dict = load_model_config(
        model_config_path=MODULE_ROOT_DIR / "model-config.toml"
    )
    train_model_and_serialize(model_config)
    predict(model_config)

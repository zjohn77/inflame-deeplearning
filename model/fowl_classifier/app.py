import json
import os
from typing import Dict

import mlflow
import tomli
import torch

from model.fowl_classifier import (
    load_model,
    MODULE_ROOT_DIR,
    run_inference,
    TrainImgClassifier,
)


def init() -> Dict[str, str]:
    """Read the config toml; make the dirs specified within."""
    with open(MODULE_ROOT_DIR / "config" / "job-config.toml", "rb") as fp:
        config = tomli.load(fp)

    # Make the output dirs if they don't already exist.
    for p in config["io"].values():
        if p.split("_")[-1] == "output":
            if not os.path.isdir(p):
                os.mkdir(p)

    return config


def train_model_and_serialize(config: dict) -> None:
    mlflow.start_run()

    # Train model and get the best model
    train_img_classifier = TrainImgClassifier()
    best_model = train_img_classifier(input_data_dir=config["io"]["training_input"])

    # Serializing it as model.pt to the specified output directory
    output_dir = config["io"]["training_output"]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    torch.save(best_model, os.path.join(output_dir, "model.pt"))

    mlflow.end_run()


def predict(config: dict) -> None:
    # Loaded trained model from expected dir, and pass a sample image to it
    model = load_model(config["io"]["training_output"])
    prediction: dict = run_inference(model, config["io"]["inference_input"])

    # Write this prediction dict to file
    output_path = os.path.join(
        config["io"]["inference_output"],
        "prediction.json",
    )
    with open(output_path, "w") as fp:
        json.dump(prediction, fp)


if __name__ == "__main__":
    job_config: dict = init()
    train_model_and_serialize(job_config)
    predict(job_config)
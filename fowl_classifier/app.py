import json
import os

import tomli
import mlflow
import torch
from fowl_classifier import TrainImgClassifier, load_model, run_inference, PROJ_ROOT_DIR


def init():
    with open(PROJ_ROOT_DIR / "config.toml", "rb") as f:
        config = tomli.load(f)

    # Make the output dirs if they don't exist already
    for p in config["io"].values():
        if p.split("_")[-1] == "output":
            if not os.path.isdir(p):
                os.mkdir(p)

    return config["io"]


def train_model_and_serialize(config: dict) -> None:
    mlflow.start_run()

    # Train model and get the best model
    train_img_classifier = TrainImgClassifier()
    best_model = train_img_classifier(input_data_dir=config["io"]["training_input"])

    # Serializing it as model.pt to the specified output directory
    output_dir = config["io"]["training_output_dir"]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    torch.save(best_model, os.path.join(output_dir, "model.pt"))

    mlflow.end_run()


def predict(config: dict) -> None:
    model = load_model(config["io"]["training_output"])
    prediction: dict = run_inference(model, config["io"]["inference_input"])

    output_path = os.path.join(
        config["io"]["inference_output"],
        "prediction.json",
    )
    with open(output_path, "w") as f:
        json.dump(prediction, f)


if __name__ == "__main__":
    io_config = init()
    train_model_and_serialize(io_config)
    predict(io_config)

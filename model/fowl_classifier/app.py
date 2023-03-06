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


class ModelTrainPredict:
    """Given a directory holding data files and kwargs of split names & the percentage of each split,
    randomly divide up the files and copy them to new subdirs created based on the split names.

    Args:
        model_config_path (PathLike or str or bytes): The path to the model config file.

    Attributes:
        _model_config (dict): Wraps pretty much all of the parameters passed to the methods.
        training_input (dict): The URI to the dir with the training and validation data.
        training_output (dict): The URI to the dir where the model.pt is written to during training run.
        inference_input (dict): The URI of a test image passed to the model during inference.
        inference_output (dict): The URI to the dir where prediction results are saved to during inference.
    """
    def __init__(self, model_config_path):
        self._model_config = self._load_model_config(model_config_path)
        io_config = self._model_config["io"]
        self.training_input = {
            "uri": io_config["training_input"],
            "uri_type": self._determine_uri_type(io_config["training_input"]),
        }
        self.training_output = {
            "uri": io_config["training_output"],
            "uri_type": self._determine_uri_type(io_config["training_output"]),
        }
        self.inference_input = {
            "uri": io_config["inference_input"],
            "uri_type": self._determine_uri_type(io_config["inference_input"]),
        }
        self.inference_output = {
            "uri": io_config["inference_output"],
            "uri_type": self._determine_uri_type(io_config["inference_output"]),
        }

    @staticmethod
    def _load_model_config(
        model_config_path: Union[os.PathLike, str, bytes]
    ) -> Dict[str, Dict[str, str]]:
        """Read the model config toml. Need to make all the paths before running any of the methods, if they
        don't already exist."""
        with open(model_config_path, "rb") as fp:
            _model_config = tomli.load(fp)

        return _model_config

    @staticmethod
    def _determine_uri_type(uri: str) -> str:
        uri_parts = uri.split("://")
        if len(uri_parts) == 1:
            return "local"
        else:
            if uri_parts[0] == "abfs":
                return "azure"
            elif uri_parts[0] == "s3":
                return "aws"
            elif uri_parts[0] == "gs":
                return "google_cloud"
            else:
                raise NotImplemented

    def train_model_and_serialize(self) -> str:
        mlflow.start_run()

        # Train model and get the best model
        train_img_classifier = TrainImgClassifier()
        if self.training_input["uri_type"] == "local":
            best_model = train_img_classifier(input_data_dir=self.training_input["uri"])
        else:
            raise NotImplemented

        # Serializing it as model.pt to the specified output directory
        if self.training_output["uri_type"] == "local":
            output_dir: str = self.training_output["uri"]
        else:
            raise NotImplemented

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        torch.save(best_model, os.path.join(output_dir, "model.pt"))

        mlflow.end_run()

        return output_dir

    def predict(self, model_dir: str, write_to_file=False) -> Dict[str, str]:
        # Loaded trained model from model_dir, and pass a sample image to it
        if self.inference_input["uri_type"] == "local":
            prediction: dict = run_inference(
                model=load_model(model_dir),
                inference_input_file=self.inference_input["uri"],
            )
        else:
            raise NotImplemented

        # Write this prediction dict to file
        if write_to_file:
            if self.inference_output["uri_type"] == "local":
                output_path = os.path.join(
                    self.inference_output["uri"],
                    "prediction.json",
                )
            else:
                raise NotImplemented

            with open(output_path, "w") as fp:
                json.dump(prediction, fp)

        return prediction


if __name__ == "__main__":
    mtp = ModelTrainPredict(model_config_path=MODULE_ROOT_DIR / "model-config.toml")
    training_output_dir = mtp.train_model_and_serialize()
    predicted_label_with_prob = mtp.predict(training_output_dir)
    print(predicted_label_with_prob)

import logging
import os
from dataclasses import dataclass
from typing import Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


@dataclass
class ModelDirStructure:
    root_dir: Union[os.PathLike, str] = os.getenv("AZUREML_MODEL_DIR")
    training_input: Union[os.PathLike, str] = os.path.join(root_dir, "training_input")
    inference_input: Union[os.PathLike, str] = os.path.join(root_dir, "inference_input")
    training_output: Union[os.PathLike, str] = os.path.join(root_dir, "training_output")
    inference_output: Union[os.PathLike, str] = os.path.join(
        root_dir, "inference_output"
    )


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory.
    AZUREML_MODEL_DIR is an environment variable created during deployment; it is the path to the model folder.
    It is the path to the model folder (~/.azureml-models/$MODEL_NAME/$VERSION)
    """
    global model

    # deserialize the model file back into a model
    model = torch.load(
        os.path.join(ModelDirStructure().training_output, "model.pt"),
        map_location=lambda storage, loc: storage,
    )
    logging.info("Init complete")


def preprocess(image_file):
    """Preprocess the input image."""
    image = Image.open(image_file)

    data_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = data_transforms(image).float()
    image = torch.tensor(image)
    image = image.unsqueeze(0)

    return image.numpy()


def run():
    """get prediction"""
    sample_image_file = os.path.join(
        ModelDirStructure().inference_input, "test_img.jpg"
    )
    plt.imshow(Image.open(sample_image_file))
    input_data = torch.tensor(preprocess(sample_image_file))

    with torch.no_grad():
        output = model(input_data)
        classes = ["chicken", "turkey"]
        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(output).numpy()[0]
        index = torch.argmax(output, 1)

    result = {
        "label": classes[index],
        "probability": str(pred_probs[index]),
    }

    return result


if __name__ == "__main__":
    init()
    run()

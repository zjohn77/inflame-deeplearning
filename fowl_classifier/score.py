import json
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


def load_model(config):
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory.
    AML_MODEL_DIR is an environment variable created during deployment; it is the path to the model folder.
    It is the path to the model folder (~/.azureml-models/$MODEL_NAME/$VERSION)
    """
    # deserialize the model file back into a model
    model = torch.load(
        os.path.join(config["io"]["training_output"], "model.pt"),
        map_location=lambda storage, loc: storage,
    )

    return model


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


def run_inference(model, config):
    """get prediction"""
    sample_image_file = config["io"]["inference_input"]
    input_data = torch.tensor(preprocess(sample_image_file))

    with torch.no_grad():
        output = model(input_data)
        classes = ["chicken", "turkey"]
        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(output).numpy()[0]
        index = torch.argmax(output, 1)

    return {
        "label": classes[index],
        "probability": str(pred_probs[index]),
    }

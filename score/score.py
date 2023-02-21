import json
import logging
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model

    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (~/.azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pt")

    # deserialize the model file back into a model
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    logging.info("Init complete")


def preprocess(image_file):
    """Preprocess the input image."""
    image = Image.open(image_file)

    data_transforms = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ]
    )

    image = data_transforms(image).float()
    image = torch.tensor(image)
    image = image.unsqueeze(0)

    return image.numpy()


def run(input_data):
    """get prediction"""
    input_data = torch.tensor(json.load(input_data)["data"])

    with torch.no_grad():
        output = model(input_data)
        classes = ["chicken", "turkey"]
        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(output).numpy()[0]
        index = torch.argmax(output, 1)

    result = {
        "label":       classes[index],
        "probability": str(pred_probs[index]),
    }

    return result


if __name__ == "__main__":
    init()
    sample_image_file = "test_img.jpg"
    plt.imshow(Image.open(sample_image_file))
    image_data = preprocess(sample_image_file)

    run(input_data)

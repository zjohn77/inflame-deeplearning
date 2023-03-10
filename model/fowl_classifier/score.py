import os
from typing import Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


def load_model(training_output_dir: Union[os.PathLike, str]):
    # deserialize the model file back into a model
    model = torch.load(
        os.path.join(training_output_dir, "model.pt"),
        map_location=lambda storage, loc: storage,
    )

    return model


def preprocess(image_file: Union[os.PathLike, str]):
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


def run_inference(model, inference_input_file: Union[os.PathLike, str]):
    """get prediction"""
    input_data = torch.tensor(preprocess(inference_input_file))

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

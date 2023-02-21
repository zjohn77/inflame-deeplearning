import argparse
import copy
import os
import time
from dataclasses import dataclass
from typing import Union
from urllib.request import urlretrieve
from zipfile import ZipFile

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


@dataclass
class ModelDirStructure:
    root_dir: Union[os.PathLike, str] = os.getenv("AZUREML_MODEL_DIR")
    training_input: Union[os.PathLike, str] = os.path.join(root_dir, "training_input")
    inference_input: Union[os.PathLike, str] = os.path.join(root_dir, "training_input")
    training_output: Union[os.PathLike, str] = os.path.join(root_dir, "training_output")
    inference_output: Union[os.PathLike, str] = os.path.join(root_dir, "inference_output")


def download_data(
    tmp_data_download_path: Union[os.PathLike, str]
) -> Union[os.PathLike, str]:
    """Download and extract the training data."""
    urlretrieve(
        "https://azuremlexamples.blob.core.windows.net/datasets/fowl_data.zip",
        filename=tmp_data_download_path,
    )

    with ZipFile(tmp_data_download_path) as z:
        z.extractall()
        print("finished extracting")
        downloaded_data_dir = z.namelist()[0]

    # clean up zip file
    os.remove(tmp_data_download_path)

    return downloaded_data_dir


def _load_data(data_dir: Union[os.PathLike, str]):
    """Make pytorch data loaders."""
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    input_data_dir,
):
    """Iterate by epoch, phase, and then each data batch."""
    since = time.time()

    # Call the previously defined data loader function
    dataloaders, dataset_sizes, class_names = _load_data(input_data_dir)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}\n", "-" * 10)
        for phase in ["train", "val"]:  # Each epoch has a training and validation phase
            if phase == "train":
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += np.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print("{} Loss: {:.3f} Acc: {:.3f}\n".format(phase, epoch_loss, epoch_acc))

            # deep copy the model if a better one has been found
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    mlflow.log_metric("time_it_took_to_train", time_elapsed)
    mlflow.log_metric("best_val_acc", np.float(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def fine_tune_model(
    input_data_dir: os.PathLike,
    num_epochs: int,
    learning_rate: float,
    momentum: float,
):
    """Load a pretrained model and reset the final fully connected layer. Return the best model.
    """
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)  # only 2 classes to predict: turkey or chicken
    model_ft = model_ft.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = SGD(model_ft.parameters(), lr=learning_rate, momentum=momentum)

    model = train_model(
        model=model_ft,
        criterion=criterion,
        optimizer=optimizer_ft,
        scheduler=lr_scheduler.StepLR(
            optimizer_ft, step_size=7, gamma=0.1
        ),  # Decay LR by a factor of 0.1 every 7 epochs
        num_epochs=num_epochs,
        input_data_dir=input_data_dir,
    )

    return model


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default=ModelDirStructure().training_output,
        help="output directory",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="learning rate",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum",
    )
    args = parser.parse_args()

    # Start MLflow
    mlflow.start_run()

    # Fit model; serializing it as model.pt to the specified output directory
    model = fine_tune_model(
        input_data_dir=download_data(
            os.path.join(ModelDirStructure().training_input, "fowl_data.zip")
        ),
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model, os.path.join(args.output_dir, "model.pt"))

    mlflow.end_run()


if __name__ == "__main__":
    global DEVICE
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cli_main()

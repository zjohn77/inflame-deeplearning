import argparse
import copy
import os
from typing import Union

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
from torchvision import models

from fowl_classifier.utils import load_data


class TrainImgClassifier:
    """Iterate by epoch, phase, and then each data batch. Return the best model."""

    def __init__(
        self,
        criterion=nn.CrossEntropyLoss(),
        num_epochs: int = 1,
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
    ):
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_epochs = num_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __call__(self, input_data_dir: Union[os.PathLike, str]):
        dataloaders, dataset_sizes, class_names = load_data(input_data_dir)

        # Load pretrained model; reset the final fully connected layer to 2 classes to predict: chicken or turkey
        model_ft = models.resnet18(pretrained=True)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, 2)
        model = model_ft.to(self.device)
        optimizer = SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
        )
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=7,
            gamma=0.1,
        )

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch}/{self.num_epochs - 1}\n", "-" * 10)
            for phase in [
                "train",
                "val",
            ]:  # Each epoch has a training and validation phase
                if phase == "train":
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                print(
                    "{} Loss: {:.3f} Acc: {:.3f}\n".format(phase, epoch_loss, epoch_acc)
                )

                # deep copy the model if a better one has been found
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        mlflow.log_metric("best_val_acc", np.float(best_acc))
        model.load_state_dict(best_model_wts)  # load best model weights

        return model


def cli_main(config):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config["io"]["training_output"],
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
    train_img_classifier = TrainImgClassifier(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
    )
    best_model = train_img_classifier(input_data_dir=config["io"]["training_input"])
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(best_model, os.path.join(args.output_dir, "model.pt"))

    mlflow.end_run()

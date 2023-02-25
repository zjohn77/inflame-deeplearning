import argparse
import copy
import os
import time
from typing import Union

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
from torchvision import models

from fowl_classifier.utils import load_data


class ImageRecogModel:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        num_epochs: int,
        learning_rate: float,
        momentum: float,
        input_data_dir: Union[os.PathLike, str],
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.input_data_dir = input_data_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _train_model(self):
        """Iterate by epoch, phase, and then each data batch."""
        since = time.time()

        dataloaders, dataset_sizes, class_names = load_data(self.input_data_dir)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch}/{self.num_epochs - 1}\n", "-" * 10)
            for phase in [
                "train",
                "val",
            ]:  # Each epoch has a training and validation phase
                if phase == "train":
                    self.scheduler.step()
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

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
                    best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - since
        mlflow.log_metric("time_it_took_to_train", time_elapsed)
        mlflow.log_metric("best_val_acc", np.float(best_acc))

        self.model.load_state_dict(best_model_wts)  # load best model weights

        return self.model

    def fine_tune_model(self):
        """Load a pretrained model and reset the final fully connected layer. Return the best model."""
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(
            num_ftrs, 2
        )  # only 2 classes to predict: turkey or chicken
        model_ft = model_ft.to(self.device)

        criterion =
        optimizer_ft = SGD(
            model_ft.parameters(), lr=self.learning_rate, momentum=self.momentum
        )

        self.model = model_ft
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer_ft
        self.scheduler = lr_scheduler.StepLR(
                optimizer_ft, step_size=7, gamma=0.1,
            )
        self.num_epochs =

        model = self._train_model(
            model=model_ft,
            criterion=criterion,
            optimizer=optimizer_ft,
            scheduler=,  # Decay LR by a factor of 0.1 every 7 epochs
            num_epochs=self.num_epochs,
            input_data_dir=self.input_data_dir,
        )

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
    model = fine_tune_model(
        input_data_dir=config["io"]["training_input"],
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model, os.path.join(args.output_dir, "model.pt"))

    mlflow.end_run()

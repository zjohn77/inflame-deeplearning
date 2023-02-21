from __future__ import division, print_function

import argparse
import copy
import os
import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


def download_data():
    """Download and extract the training data."""
    import urllib
    from zipfile import ZipFile

    # download data
    data_file = "./fowl_data.zip"
    download_url = ("https://azuremlexamples.blob.core.windows.net/datasets/fowl_data.zip")
    urllib.request.urlretrieve(download_url, filename=data_file)

    # extract files
    with ZipFile(data_file, "r") as zip:
        print("extracting files...")
        zip.extractall()
        print("finished extracting")
        data_dir = zip.namelist()[0]

    # delete zip file
    os.remove(data_file)

    return data_dir


def load_data(data_dir):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        "train":  transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ]
        ), "val": transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ]
        ),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val"]}
    dataloaders = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=4, shuffle=True, num_workers=4
    ) for x in ["train", "val"]}
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names


def train_model(model, criterion, optimizer, scheduler, num_epochs, data_dir):
    # load training/validation data
    dataloaders, dataset_sizes, class_names = load_data(data_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += np.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}\n".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # log the best val accuracy to AML run
            mlflow.log_metric("best_val_acc", np.float(best_acc))

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def fine_tune_model(num_epochs, data_dir, learning_rate, momentum):
    """Load a pretrained model and reset the final fully connected layer."""
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)  # only 2 classes to predict

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(
        model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs, data_dir
    )

    return model


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


def main():
    print("Torch version:", torch.__version__)

    # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="number of epochs to train"
    )
    parser.add_argument("--output_dir", type=str, default="./output", help="output directory")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    args = parser.parse_args()

    # Start Run
    mlflow.start_run()

    data_dir = download_data()
    print("data directory is: " + data_dir)
    model = fine_tune_model(
        args.num_epochs, data_dir, args.learning_rate, args.momentum
    )
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model, os.path.join(args.output_dir, "model.pt"))

    mlflow.end_run()

    sample_image_file = "test_img.jpg"
    plt.imshow(Image.open(sample_image_file))
    image_data = preprocess(sample_image_file)


if __name__ == "__main__":
    main()

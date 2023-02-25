import os
from typing import Union
from urllib.request import urlretrieve
from zipfile import ZipFile

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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


def load_data(data_dir: Union[os.PathLike, str]):
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

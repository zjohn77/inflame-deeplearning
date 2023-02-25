import os
from dataclasses import dataclass
from typing import Union
from urllib.request import urlretrieve
from zipfile import ZipFile


@dataclass
class ModelDirStructure:
    root_dir: Union[os.PathLike, str] = "/Users/jj/.aml-models"
    training_input: Union[os.PathLike, str] = os.path.join(root_dir, "training_input")
    inference_input: Union[os.PathLike, str] = os.path.join(root_dir, "inference_input")
    training_output: Union[os.PathLike, str] = os.path.join(root_dir, "training_output")
    inference_output: Union[os.PathLike, str] = os.path.join(
        root_dir, "inference_output"
    )


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

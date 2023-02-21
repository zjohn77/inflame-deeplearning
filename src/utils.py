from urllib.request import urlretrieve
from zipfile import ZipFile
from typing import Union
import os


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

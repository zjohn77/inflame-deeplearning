import logging
import os
from typing import Union

from azure.storage.blob import BlobClient


class Access:
    def __init__(self, job_config: dict):
        self.logger = logging.getLogger(__file__)
        self.container_name = job_config["workspace"]["storage_container_name"]

    def download(
        self,
        blob_name: str,
        local_download_dir: Union[os.PathLike, str, bytes],
    ):
        try:
            blob_client = BlobClient.from_connection_string(
                conn_str=os.getenv("AZURE_STORAGE_ACCOUNT_CONNECTION_STRING"),
                container_name=self.container_name,
                blob_name=blob_name,
            )
        except AttributeError:
            raise "Connection string environment variable not set properly."

        local_download_path = os.path.join(local_download_dir, blob_name)
        with open(local_download_path, "wb") as fp:
            blob_data = blob_client.download_blob()
            blob_data.readinto(fp)

        self.logger.info(
            f"Dowloaded blob from Azure Storage Container: {self.container_name}\n"
            "to: {local_download_path}"
        )

import logging
import os
from typing import Union
from urllib.parse import urlparse

from adlfs import AzureBlobFileSystem


class FsAccess:
    def __init__(self, job_config: dict):
        self.logger = logging.getLogger(__file__)

        storage_account_conn_str = os.getenv("AZURE_STORAGE_ACCOUNT_CONNECTION_STRING")
        if not storage_account_conn_str:
            raise "Connection string environment variable not set properly."
        self.fs = AzureBlobFileSystem(connection_string=storage_account_conn_str)

        self.uri = job_config["workspace"]["storage_uri"]
        parsed_uri = urlparse(self.uri)
        self.container_name = parsed_uri.netloc
        self.blob_name = parsed_uri.path.lstrip("/")

    def list_container(self):
        return self.fs.ls(self.container_name)

    def download(
        self,
        local_download_dir: Union[os.PathLike, str, bytes],
    ):
        local_download_path = os.path.join(local_download_dir, self.blob_name)
        self.fs.get(
            rpath=self.uri,
            lpath=local_download_path,
            recursive=False,
        )
        self.logger.info(
            f"Dowloaded blob from Azure Storage Container: {self.container_name}\n"
            f"to: {local_download_path}"
        )

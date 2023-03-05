from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.storage.blob import BlobServiceClient


# noinspection PyTypeChecker
class Access:
    def __init__(self):
        # Most likely need to get credentials from Azure CLI via: az login
        try:
            self.credential = DefaultAzureCredential()
            self.credential.get_token("https://management.azure.com/.default")
        except Exception:
            self.credential = InteractiveBrowserCredential()

    def download(self):
        blob_service_client = BlobServiceClient(
            account_url=self.config["workspace"]["storage_account_url"],
            credential=self.credential,
        )

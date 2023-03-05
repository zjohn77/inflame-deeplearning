import logging

import tomli
from azure.ai.ml import command, MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

from model.fowl_classifier import MODULE_ROOT_DIR


# noinspection PyTypeChecker
class RunTrainingJob:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu

        try:
            with open(MODULE_ROOT_DIR / "config" / "job-config.toml", mode="rb") as fp:
                self.config = tomli.load(fp)
        except tomli.TOMLDecodeError:
            logger.error("Invalid toml config.")

        # Most likely need to get credentials from Azure CLI via: az login
        try:
            self.credential = DefaultAzureCredential()
            self.credential.get_token("https://management.azure.com/.default")
        except Exception as exc:
            logger.error(exc)
            self.credential = InteractiveBrowserCredential()

        self.ml_client = MLClient(
            subscription_id=self.config["workspace"]["subscription_id"],
            resource_group_name=self.config["workspace"]["resource_group_name"],
            workspace_name=self.config["workspace"]["workspace_name"],
            credential=self.credential,
        )

    def __create_compute_resource__(self, compute_config):
        try:
            # Try to get existing compute resource with this name.
            return self.ml_client.compute.get(compute_config["name"])
        except Exception as exc:
            logger.error(f"Error: {exc}. Creating a new compute target...")
            return self.ml_client.begin_create_or_update(
                AmlCompute(
                    name=compute_config["name"],
                    type="amlcompute",
                    size=compute_config["size"],
                    min_instances=0,
                    max_instances=4,
                    idle_time_before_scale_down=60,
                    tier="Dedicated",
                )
            ).result()

    def __call__(self):
        """Submit job to the created compute instance."""
        if self.use_gpu:
            cluster = self.__create_compute_resource__(self.config["compute.gpu"])
        else:
            cluster = self.__create_compute_resource__(self.config["compute.cpu"])

        return self.ml_client.jobs.create_or_update(
            command(
                display_name=self.config["job"]["display_name"],
                inputs=dict(
                    num_epochs=self.config["hyperparam"]["num_epochs"],
                    learning_rate=self.config["hyperparam"]["learning_rate"],
                    momentum=self.config["hyperparam"]["momentum"],
                ),
                code=self.config["io"]["code_module_dir"],
                command=self.config["job"]["command"],
                environment=self.config["compute.gpu"]["environment"],
                compute=cluster.name,
            )
        )


if __name__ == "__main__":
    logger = logging.getLogger(__file__)

    submitted_job = RunTrainingJob(use_gpu=True)()
    logger.info(f"The submitted job object on AML: {submitted_job}")

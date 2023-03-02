import logging

import tomli
from azure.ai.ml import command, MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential


def create_aml_client(config):
    # Most likely need to get credentials from Azure CLI via: az login
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as exc:
        logger.error(exc)
        credential = InteractiveBrowserCredential()

    # noinspection PyTypeChecker
    ml_client = MLClient(
        subscription_id=config["workspace"]["subscription_id"],
        resource_group_name=config["workspace"]["resource_group_name"],
        workspace_name=config["workspace"]["workspace_name"],
        credential=credential,
    )

    return ml_client


def create_compute_instance(ml_client, config):
    """TODO: add conditionals for switching between gpu and cpu."""
    try:
        # Try to get existing compute resource with this name
        return ml_client.compute.get(config["compute"]["compute_target"])
    except Exception as exc:
        logger.error(f"Error: {exc}. Creating a new compute target...")
        gpu_cluster = AmlCompute(
            name="gpu-cluster",
            type="amlcompute",
            size=config["compute"]["size"],
            min_instances=0,
            max_instances=4,
            idle_time_before_scale_down=60,
            tier="Dedicated",
        )
        return ml_client.begin_create_or_update(gpu_cluster).result()


def create_and_submit_job(ml_client, config):
    """Submit job to the created compute instance or local"""
    job = command(
        code="./fowl_classifier",
        command="python app.py",
        inputs=dict(
            num_epochs=1,
            learning_rate=0.001,
            momentum=0.9,
            output_dir="./output",
        ),
        environment=config["compute"]["environment"],
        compute=config["compute"]["compute_target"],
        experiment_name="turkey-or-chicken-experiment",
        display_name=config["command"]["display_name"],
        description=config["command"]["description"],
    )

    return ml_client.jobs.create_or_update(job)


if __name__ == "__main__":
    logger = logging.getLogger(__file__)

    LOCAL_RUN = True
    if LOCAL_RUN:
        with open("configs/local-run-config.toml", mode="rb") as fp:
            local_run_config = tomli.load(fp)

        aml_client = create_aml_client(local_run_config)

        response = create_and_submit_job(aml_client, local_run_config)
        logger.info(f"Submitted job with response: {response}")
    else:
        with open("configs/remote-job-config.toml", mode="rb") as fp:
            remote_job_config = tomli.load(fp)

        aml_client = create_aml_client(remote_job_config)

        cluster = create_compute_instance(aml_client, remote_job_config)
        logger.info(
            f"Created AMLCompute called {cluster.name} with size: {cluster.size}."
        )

        response = create_and_submit_job(aml_client, remote_job_config)
        logger.info(f"Submitted job with response: {response}")

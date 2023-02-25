import logging

import tomli
from azure.ai.ml import command, MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential


def create_aml_client(config):
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")

    except Exception as exc:
        logger.info(exc)
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
    compute_target = config["compute"]["compute_target"]
    try:
        return ml_client.compute.get(compute_target)
    except Exception:
        logger.info("Creating a new compute target...")
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
    """Submit job to the created compute instance"""
    job = command(
        code="./src",
        command=(
            "python pytorch_train.py --num_epochs ${{inputs.num_epochs}} --output_dir"
            " ${{inputs.output_dir}}"
        ),
        inputs=dict(
            num_epochs=1,
            learning_rate=0.001,
            momentum=0.9,
            output_dir="./output",
        ),
        environment=config["compute"]["environment"],
        compute=config["compute"]["compute_target"] or "local",
        display_name="new-bird-job",
        description="The description of the experiment",
    )

    return ml_client.jobs.create_or_update(job)


if __name__ == "__main__":
    logger = logging.getLogger(__file__)

    with open("config.toml", mode="rb") as fp:
        config = tomli.load(fp)

    aml_client = create_aml_client(config)

    cluster = create_compute_instance(aml_client, config)
    print(
        f"AMLCompute with name {cluster.name} was created, with {cluster.size} compute size."
    )

    response = create_and_submit_job(aml_client, config)
    print(response)

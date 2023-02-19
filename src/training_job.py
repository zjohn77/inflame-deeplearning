from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential
import os

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.getenv("SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("RESOURCE_GROUP"),
    workspace_name=os.getenv("AML_WORKSPACE_NAME"),
)

GPU_COMPUTE_TAGET = "gpu-cluster"
CURATED_ENV_NAME = "AzureML-pytorch-1.9-ubuntu18.04-py37-cuda11-gpu@latest"

try:
    # let's see if the compute target already exists
    gpu_cluster = ml_client.compute.get(GPU_COMPUTE_TAGET)
    print(
        f"You already have a cluster named {GPU_COMPUTE_TAGET}, we'll reuse it as is."
    )
except Exception:
    print("Creating a new gpu compute target...")

    # Let's create the Azure ML compute object with the intended parameters
    gpu_cluster = AmlCompute(
        # Name assigned to the compute cluster
        name="gpu-cluster",
        # Azure ML Compute is the on-demand VM service
        type="amlcompute",
        # VM Family
        size="STANDARD_NC6",
        # Minimum running nodes when there is no job running
        min_instances=0,
        # Nodes in cluster
        max_instances=4,
        # How many seconds will the node running after the job termination
        idle_time_before_scale_down=180,
        # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
        tier="Dedicated",
    )

    # Now, we pass the object to MLClient's create_or_update method
    gpu_cluster = ml_client.begin_create_or_update(gpu_cluster).result()

print(
    f"AMLCompute with name {gpu_cluster.name} is created, the compute size is {gpu_cluster.size}"
)

job = command(
    inputs=dict(
        num_epochs=30, learning_rate=0.001, momentum=0.9, output_dir="./outputs"
    ),
    compute=GPU_COMPUTE_TAGET,
    environment=CURATED_ENV_NAME,
    code="./trainer/",  # location of source code
    command="python pytorch_train.py --num_epochs ${{inputs.num_epochs}} --output_dir ${{inputs.output_dir}}",
    experiment_name="pytorch-birds",
    display_name="pytorch-birds-image",
)

ml_client.jobs.create_or_update(job)

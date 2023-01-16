import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GCP_PROJ = "my-workload-id-federation"
BUCKET = "euclid-garden-prod"
# DATA_SAVE_PATH = BUCKET / "data" / "iris.csv"
# MODEL_SAVE_PATH = BUCKET / "model_artifacts" / "nn_iris_1.pt"

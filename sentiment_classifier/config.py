from pathlib import Path

HIDDEN_UNITS = 10
LEARNING_RATE = 0.1
CLASS_VOCAB = ["setosa", "versicolor", "virginica"]
NUM_EPOCHS = 4

PROJ_ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_SAVE_PATH = PROJ_ROOT_DIR / "data" / "iris.csv"
MODEL_SAVE_PATH = PROJ_ROOT_DIR / "model_artifacts" / "nn_iris_1.pt"

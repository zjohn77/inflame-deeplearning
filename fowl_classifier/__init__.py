from pathlib import Path

from .train import TrainImgClassifier
from .score import load_model, run_inference

PROJ_ROOT_DIR = Path(__file__).parents[1].resolve()

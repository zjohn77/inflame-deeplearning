from pathlib import Path

from .score import load_model, run_inference
from .train import TrainImgClassifier

MODULE_ROOT_DIR = Path(__file__).parents[0].resolve()

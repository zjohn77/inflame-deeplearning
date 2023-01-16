from transformers import TrainingArguments
from config import PROJ_ROOT_DIR

training_args = TrainingArguments(
    output_dir=PROJ_ROOT_DIR / "model_artifacts",
    learning_rate=1e-3,
    num_train_epochs=10
)

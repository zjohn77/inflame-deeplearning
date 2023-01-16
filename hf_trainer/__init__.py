from transformers import AutoModelForSequenceClassification

from config import MODEL_CHECKPOINT
from .hyperparams import training_args
from .preprocess import tokenize

model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=2)


def training_loop():
    pass

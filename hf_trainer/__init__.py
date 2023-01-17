from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    Trainer,
)

from config import MODEL_CHECKPOINT
from .hyperparams import training_args
from .preprocess import load_data, tokenize_function


def get_pretrained_model(model_checkpoint):
    return AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=2,
    )


def training_loop():
    data = load_data().map(tokenize_function, batched=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    trainer = Trainer(
        model=get_pretrained_model(MODEL_CHECKPOINT),
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        tokenizer=tokenizer,
    )
    trainer.train()

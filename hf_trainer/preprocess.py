from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from config import MODEL_CHECKPOINT

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
DATA_COLLATOR = DataCollatorWithPadding(tokenizer=TOKENIZER)


def load_data():
    """
    MRPC subtask of the GLUE task.
    """
    raw_datasets = load_dataset("glue", "mrpc")
    return raw_datasets


def tokenize_function(example):
    """
    Pairs of args to TOKENIZER because the task is determining if one sentence is a paraphrase
    of the other.
    """
    return TOKENIZER(
        example["sentence1"],
        example["sentence2"],
        truncation=True,
    )


def tokenize(input_datasets):
    tokenized_datasets = input_datasets.map(
        tokenize_function,
        batched=True,
    )
    return tokenized_datasets

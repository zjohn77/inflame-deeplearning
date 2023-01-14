import pandas as pd
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer

from trainer import (
    PRETRAINED_MODEL_NAME,
    MAX_SEQ_LENGTH,
    TARGET_LABELS,
    TRAIN_DATA,
    TEST_DATA,
)


def preprocess_function(examples):
    tokenizer = AutoTokenizer.from_pretrained(
        PRETRAINED_MODEL_NAME,
        use_fast=True,
    )

    result = tokenizer(
        *((examples["text"],)),
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        truncation=True
    )

    label_to_id = TARGET_LABELS
    if label_to_id is not None and "label" in examples:
        result["label"] = [label_to_id[l] for l in examples["label"]]

    return result


def load_data(args):
    """Loads the data into two different data loaders. (Train, Test)

    Args:
        args: arguments passed to the python script
    """
    # dataset loading repeated here to make this cell idempotent
    # since we are over-writing datasets variable
    df_train = pd.read_csv(TRAIN_DATA)
    df_test = pd.read_csv(TEST_DATA)

    dataset = DatasetDict(
        {"train": Dataset.from_pandas(df_train), "test": Dataset.from_pandas(df_test)}
    )

    dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=True)

    train_dataset, test_dataset = dataset["train"], dataset["test"]

    return train_dataset, test_dataset

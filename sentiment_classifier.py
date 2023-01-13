import os
import datetime
import pandas as pd

from google.cloud import storage

from transformers import AutoTokenizer
from datasets import load_dataset, load_metric, ReadInstruction, DatasetDict, Dataset
from transformers import AutoModelForSequenceClassification


PRETRAINED_MODEL_NAME = 'bert-base-cased'
MAX_SEQ_LENGTH = 128
TARGET_LABELS = {"leisure": 0, "exercise":1, "enjoy_the_moment":2, "affection":3,"achievement":4, "nature":5, "bonding":6}
TRAIN_DATA = "gs://cloud-samples-data/ai-platform-unified/datasets/text/happydb/happydb_train.csv"
TEST_DATA = "gs://cloud-samples-data/ai-platform-unified/datasets/text/happydb/happydb_test.csv"



def create(num_labels):
    """create the model by loading a pretrained model or define your
    own

    Args:
      num_labels: number of target labels
    """
    # Create the model, loss function, and optimizer
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=num_labels
    )

    return model





def preprocess_function(examples):
    tokenizer = AutoTokenizer.from_pretrained(
        PRETRAINED_MODEL_NAME,
        use_fast=True,
    )

    result = tokenizer(*(
        (examples['text'],)
    ),
                       padding='max_length',
                       max_length=MAX_SEQ_LENGTH,
                       truncation=True)

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

    dataset = DatasetDict({"train": Dataset.from_pandas(df_train), "test": Dataset.from_pandas(df_test)})

    dataset = dataset.map(preprocess_function,
                          batched=True,
                          load_from_cache_file=True)

    train_dataset, test_dataset = dataset["train"], dataset["test"]

    return train_dataset, test_dataset


def save_model(args):
    """Saves the model to Google Cloud Storage or local file system

    Args:
      args: contains name for saved model.
    """
    scheme = 'gs://'
    if args.job_dir.startswith(scheme):
        job_dir = args.job_dir.split("/")
        bucket_name = job_dir[2]
        object_prefix = "/".join(job_dir[3:]).rstrip("/")

        if object_prefix:
            model_path = '{}/{}'.format(object_prefix, args.model_name)
        else:
            model_path = '{}'.format(args.model_name)

        bucket = storage.Client().bucket(bucket_name)
        local_path = os.path.join("/tmp", args.model_name)
        files = [f for f in os.listdir(local_path) if os.path.isfile(os.path.join(local_path, f))]
        for file in files:
            local_file = os.path.join(local_path, file)
            blob = bucket.blob("/".join([model_path, file]))
            blob.upload_from_filename(local_file)
        print(f"Saved model files in gs://{bucket_name}/{model_path}")
    else:
        print(f"Saved model files at {os.path.join('/tmp', args.model_name)}")
        print(f"To save model files in GCS bucket, please specify job_dir starting with gs://")
from datasets import load_dataset


def load_data():
    """MRPC subtask of the GLUE task. Returns the raw datasets.
    """
    return load_dataset("glue", "mrpc")


def tokenize_function(example, tokenizer):
    """Pairs of args to tokenizer because the task is determining if one sentence is a paraphrase
    of the other.
    """
    return tokenizer(
        example["sentence1"],
        example["sentence2"],
        truncation=True,
    )

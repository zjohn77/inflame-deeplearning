import math
import os
import random
import warnings
from pathlib import Path
from typing import List, Union, Iterable


def train_dev_eval_split(
    datadir: Path,
    fracs: Iterable[float],
) -> List[Union[List[Path], None]]:
    """
    :param datadir: Dir containing files to be split into train, dev, eval, etc.
    :param fracs: Fractions of splits to be produced

    :return: A list that holds, for example, train-val-test splits of inputs.

    Examples
    --------
    >>> train_dev_eval_split(datadir=Path("path/to/data"), fracs=[0.8, 0.1, 0.1])
    """

    # 0. The list of data filenames is the population we're sampling from
    data_filenames = [
        filename
        for filename in datadir.iterdir()
        if not str(filename).endswith(".DS_Store")
    ]
    population_size: int = len(data_filenames)
    if population_size == 0:
        raise ValueError("At least one file must be in datadir.")

    assert (
        math.isclose(sum(fracs), 1) and sum(fracs) <= 1
    )  # percentages must sum up to 100%

    # 1. Figure out samples sizes of the splits
    sample_sizes = [0] * len(fracs)
    for i, frac in enumerate(fracs):
        if frac < 0 or frac > 1:
            raise ValueError(f"The {i}th percentage is not between 0 and 100%")

        sample_sizes[i] = int(math.floor(frac * population_size))

    # Assign to the training set (1st split) any leftovers resulting from rounding down
    sample_sizes[0] += population_size - sum(sample_sizes)

    # 2. Shuffle and split filenames based on the sample sizes from before
    random.shuffle(data_filenames)

    splits: List[Union[List[Path], None]] = [None] * len(sample_sizes)
    offset = 0
    for i, sample_size in enumerate(sample_sizes):
        if sample_size == 0:
            warnings.warn(f"The {i}th sample size is 0; an empty split will result.")
        splits[i] = data_filenames[offset : offset + sample_size]
        offset += sample_size

    return splits


def mv_files_to_dir(src_paths: List[Path], dest_dir: Path):
    """
    :param src_paths: File paths to be moved to a new destination
    :param dest_dir: The destination Path; if this folder doesn't yet exist, it'll be created

    :return: The number of files in the dir where the files were moved to (a good diagnostic to log)
    """
    try:
        dest_dir.mkdir()
    except FileExistsError:
        raise FileExistsError(f"Aborting because {dest_dir} already exists.")

    for src_path in src_paths:
        dest_path = dest_dir / src_path.name
        os.rename(str(src_path.absolute()), str(dest_path.absolute()))

    return sum(len(files) for _, _, files in os.walk(dest_dir))

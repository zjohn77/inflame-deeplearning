import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import List, Union, Dict

from artifact import ParseKwargs
from .random_data_split import train_dev_eval_split, mv_files_to_dir


def split_datafiles(
    datadir: Path,
    **splits: Union[None, Dict[str, float]],
):
    """Given a directory holding data files and kwargs of split names & frac of each split,
    randomly divide up the files and move them to new subdirs created based on the split names.

    :param datadir: Dir containing list of files to be split into train, dev, eval, etc.
    :param splits: Kwargs of split names & frac of each split

    Write to datadir/data_splits_index.json something like the following:
    {
        "train": ['/pthname1', '/pthname2'],
        "val": ['/pthname3']
    }

    Examples
    --------
    % python -m artifact /Users/jj/Public/data/docsum_datasets/bbcnews/data
    % python -m artifact /Users/jj/Public/data/docsum_datasets/bbcnews/data --splits train=0.8 dev=0.1 eval=0.1
    """
    if not isinstance(splits, dict) or len(splits) < 2:
        splits = {"train": 0.8, "val": 0.2}  # default splits

    ordered_splits = OrderedDict(splits)
    train_dev_eval: List[List[Path]] = train_dev_eval_split(
        datadir=datadir,
        fracs=ordered_splits.values(),
    )

    split_names = ordered_splits.keys()
    for i, split_name in enumerate(split_names):
        mv_files_to_dir(
            src_paths=train_dev_eval[i],
            dest_dir=datadir / split_name,
        )

    datafiles_index: Dict[str, List[str]] = {
        split_name: list(map(lambda pth: str(pth.absolute()), filelist))
        for split_name, filelist in zip(split_names, train_dev_eval)
    }

    with open(datadir.parent / "data_splits_index.json", "w") as f:
        json.dump(datafiles_index, f)


def cli_split_datafiles():
    parser = argparse.ArgumentParser(
        description="Given a dir, randomly split files contained herein."
    )
    parser.add_argument("datadir")
    # noinspection PyTypeChecker
    parser.add_argument("--splits", nargs="+", action=ParseKwargs)
    args = parser.parse_args()

    split_datafiles(datadir=Path(args.datadir), **args.splits if args.splits else {})


if __name__ == "__main__":
    cli_split_datafiles()

import argparse
from pathlib import Path

from artifact import split_datafiles


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = float(value)


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

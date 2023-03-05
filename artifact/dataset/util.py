import argparse


def path2str(splits):
    str_path_tree = []
    for split in splits:
        str_path_branch = []
        for filepath in split:
            str_path_branch.append(str(filepath.absolute()))
        str_path_tree.append(str_path_branch)
    return str_path_tree


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = float(value)

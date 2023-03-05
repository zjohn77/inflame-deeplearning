from .cli import ParseKwargs


def path2str(splits):
    str_path_tree = []
    for split in splits:
        str_path_branch = []
        for filepath in split:
            str_path_branch.append(str(filepath.absolute()))
        str_path_tree.append(str_path_branch)
    return str_path_tree

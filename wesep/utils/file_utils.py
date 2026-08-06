"""Minimal file helpers for DAE-TSE inference."""


def read_lists(list_file):
    """Read a 1-column list / scp file."""
    lists = []
    with open(list_file, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                lists.append(line)
    return lists

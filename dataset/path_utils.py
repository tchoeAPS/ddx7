from pathlib import Path


def ensure_nested_dir(root, relative_path):
    target_dir = Path(root) / relative_path
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir

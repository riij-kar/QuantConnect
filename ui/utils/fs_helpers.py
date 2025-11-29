import os
from typing import List, Dict

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

def safe_join(base: str, *paths: str) -> str:
    """Join path components and reject attempts to escape the base directory."""
    joined = os.path.abspath(os.path.join(base, *paths))
    if not joined.startswith(os.path.abspath(base)):
        raise ValueError('Path traversal detected')
    return joined

def is_dir_path(path: str) -> bool:
    """Return True if the given relative path under DATA_ROOT is a directory."""
    abs_path = safe_join(DATA_ROOT, path)
    return os.path.isdir(abs_path)

def list_dir(path: str, max_entries: int = 200) -> List[Dict]:
    """Return list of directory contents with type info."""
    abs_path = safe_join(DATA_ROOT, path)
    entries = []
    try:
        for name in sorted(os.listdir(abs_path))[:max_entries]:
            full = os.path.join(abs_path, name)
            entries.append({
                'name': name,
                'path': os.path.relpath(full, DATA_ROOT).replace('\\', '/'),
                'is_dir': os.path.isdir(full)
            })
    except (FileNotFoundError, NotADirectoryError):
        # If path doesn't exist or isn't a directory, return empty listing
        return []
    return entries

def ensure_folder(path: str) -> str:
    """Create the folder under DATA_ROOT if it does not already exist."""
    abs_path = safe_join(DATA_ROOT, path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path

def save_csv(path: str, filename: str, content: str) -> str:
    """Persist CSV content under DATA_ROOT and return the absolute file path."""
    ensure_folder(path)
    abs_target = safe_join(DATA_ROOT, path, filename)
    with open(abs_target, 'w', newline='') as f:
        f.write(content)
    return abs_target

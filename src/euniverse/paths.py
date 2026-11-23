# MIT License — see LICENSE in the project root.

"""
paths.py — per-user cache location, the same convention on every platform.

  Linux    $XDG_CACHE_HOME or ~/.cache          -> .../euniverse/<name>
  macOS    ~/Library/Caches                      -> .../euniverse/<name>
  Windows  %LOCALAPPDATA%                        -> ...\\euniverse\\<name>
"""

import os
import sys


def cache_dir(name: str) -> str:
    """Path of the cache sub-directory *name* (not created)."""
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
    elif sys.platform == "darwin":
        base = os.path.join(home, "Library", "Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(home, ".cache")
    return os.path.join(base, "euniverse", name)

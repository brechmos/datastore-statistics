"""Utility functions for datastore statistics."""

import pathlib
import re
from typing import Union

import numpy as np


def threshold(x: float, thr: float, thr_direction: str) -> bool:
    """Apply threshold filtering with directional logic.

    Args:
        x: Value to check
        thr: Threshold value
        thr_direction: One of 'absolute', 'lower', 'higher'

    Returns:
        True if value exceeds threshold in specified direction

    Raises:
        ValueError: If thr_direction is not one of 'absolute', 'lower', 'higher'
    """
    if thr_direction == "absolute":
        return bool(np.abs(x) > thr)
    elif thr_direction == "lower":
        thr = -thr if thr < 0 else thr
        return bool(x < -thr)
    elif thr_direction == "higher":
        return bool(x > thr)
    raise ValueError(
        f"Invalid threshold direction: {thr_direction!r}. "
        f"Must be one of 'absolute', 'lower', 'higher'."
    )


def find_files(
    data_directory: Union[str, pathlib.Path], filename_regexp: str
) -> list[pathlib.Path]:
    """Find files matching a pattern in a directory.

    Args:
        data_directory: Path to search
        filename_regexp: Regex pattern for filtering filenames

    Returns:
        List of matching file paths, sorted
    """
    path = pathlib.Path(data_directory)
    filenames = [
        x for x in sorted(path.rglob("*")) if re.match(filename_regexp, str(x.name))
    ]
    return filenames

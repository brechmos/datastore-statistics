"""Lightweight outlier detection for medical image datasets."""

__version__ = "0.2.0"

from .readers import DataFile  # noqa: F401
from .report import generate_html_report  # noqa: F401
from .stats import (  # noqa: F401
    add_stats,
    compute_efc,
    compute_fber,
    compute_outliers,
    otsu_threshold,
)
from .utils import find_files  # noqa: F401

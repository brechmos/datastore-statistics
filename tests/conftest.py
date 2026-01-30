"""Pytest fixtures for datastore-statistics tests."""

import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List

import numpy as np
import pytest


@pytest.fixture
def sample_2d_image() -> np.ndarray:
    """Create a simple 2D test image."""
    np.random.seed(42)
    data: np.ndarray = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
    return data


@pytest.fixture
def sample_3d_volume() -> np.ndarray:
    """Create a simple 3D test volume."""
    np.random.seed(42)
    data: np.ndarray = np.random.randint(0, 256, size=(64, 64, 32), dtype=np.uint8)
    return data


@pytest.fixture
def blank_image() -> np.ndarray:
    """Create a blank (all zeros) image."""
    data: np.ndarray = np.zeros((64, 64), dtype=np.uint8)
    return data


@pytest.fixture
def constant_image() -> np.ndarray:
    """Create a constant value image."""
    data: np.ndarray = np.full((64, 64), 128, dtype=np.uint8)
    return data


@pytest.fixture
def bimodal_image() -> np.ndarray:
    """Create an image with clear foreground/background separation."""
    img: np.ndarray = np.zeros((64, 64), dtype=np.uint8)
    img[16:48, 16:48] = 200  # Bright square in center
    return img


@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_file_list() -> List[Dict[str, Any]]:
    """Create a sample file list with computed stats."""
    return [
        {"filename": "img1.jpg", "mean": 100, "std": 10, "efc": 0.8, "fber": 5.0},
        {"filename": "img2.jpg", "mean": 105, "std": 12, "efc": 0.82, "fber": 4.8},
        {"filename": "img3.jpg", "mean": 98, "std": 11, "efc": 0.79, "fber": 5.2},
        {
            "filename": "img4.jpg",
            "mean": 200,
            "std": 15,
            "efc": 0.9,
            "fber": 3.0,
        },  # Outlier
        {"filename": "img5.jpg", "mean": 102, "std": 9, "efc": 0.81, "fber": 4.9},
    ]

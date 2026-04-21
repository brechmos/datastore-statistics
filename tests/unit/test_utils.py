"""Unit tests for datastore_statistics.utils module."""

from pathlib import Path

import pytest

from datastore_statistics.utils import find_files, threshold


class TestThreshold:
    """Tests for threshold function."""

    def test_absolute_above_threshold(self) -> None:
        """Test absolute threshold with value above."""
        assert threshold(3.0, 2.0, "absolute")
        assert threshold(-3.0, 2.0, "absolute")

    def test_absolute_below_threshold(self) -> None:
        """Test absolute threshold with value below."""
        assert not threshold(1.5, 2.0, "absolute")
        assert not threshold(-1.5, 2.0, "absolute")

    def test_lower_direction(self) -> None:
        """Test lower direction threshold."""
        assert threshold(-3.0, 2.0, "lower") is True
        assert threshold(-1.0, 2.0, "lower") is False
        assert threshold(3.0, 2.0, "lower") is False

    def test_higher_direction(self) -> None:
        """Test higher direction threshold."""
        assert threshold(3.0, 2.0, "higher") is True
        assert threshold(1.0, 2.0, "higher") is False
        assert threshold(-3.0, 2.0, "higher") is False

    def test_invalid_direction(self) -> None:
        """Test that invalid direction raises ValueError."""
        with pytest.raises(ValueError, match="Invalid threshold direction"):
            threshold(3.0, 2.0, "invalid")


class TestFindFiles:
    """Tests for find_files function."""

    def test_find_jpeg_files(self, test_data_dir: Path) -> None:
        """Test finding JPEG files in test data directory."""
        files = find_files(str(test_data_dir), r".*\.jpeg")
        assert len(files) > 0
        assert all(str(f).endswith(".jpeg") for f in files)

    def test_find_with_complex_regex(self, test_data_dir: Path) -> None:
        """Test finding files with complex regex pattern."""
        files = find_files(str(test_data_dir), r".*\.(jpeg|jpg|png)")
        assert len(files) > 0

    def test_find_no_matches(self, test_data_dir: Path) -> None:
        """Test that non-matching pattern returns empty list."""
        files = find_files(str(test_data_dir), r".*\.xyz")
        assert len(files) == 0

    def test_find_files_sorted(self, test_data_dir: Path) -> None:
        """Test that results are sorted."""
        files = find_files(str(test_data_dir), r".*\.jpeg")
        filenames = [f.name for f in files]
        assert filenames == sorted(filenames)

    def test_nonexistent_directory(self, temp_dir: Path) -> None:
        """Test finding files in nonexistent directory."""
        files = find_files(str(temp_dir / "nonexistent"), r".*\.txt")
        assert len(files) == 0

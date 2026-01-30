"""Unit tests for datastore_statistics.readers module."""

from pathlib import Path

import numpy as np
import pytest

from datastore_statistics.readers import DataFile


class TestDataFile:
    """Tests for DataFile factory class."""

    def test_get_reader_jpeg(self, test_data_dir: Path) -> None:
        """Test getting reader for JPEG file."""
        jpeg_files = list(test_data_dir.glob("*.jpeg"))
        if not jpeg_files:
            pytest.skip("No JPEG test files available")

        reader = DataFile.get_reader(jpeg_files[0])
        assert reader is not None

    def test_get_reader_returns_data(self, test_data_dir: Path) -> None:
        """Test that reader returns numpy array data."""
        jpeg_files = list(test_data_dir.glob("*.jpeg"))
        if not jpeg_files:
            pytest.skip("No JPEG test files available")

        reader = DataFile.get_reader(jpeg_files[0])
        data = reader.get_data()

        assert isinstance(data, np.ndarray)
        assert data.ndim >= 2

    def test_get_reader_unsupported_extension(self, temp_dir: Path) -> None:
        """Test that unsupported extension raises error."""
        fake_file = temp_dir / "test.xyz"
        fake_file.write_text("fake")

        with pytest.raises(ValueError, match="No reader available for file type"):
            DataFile.get_reader(fake_file)


class TestImageIOReader:
    """Tests for ImageIO reader (JPEG, PNG)."""

    def test_read_jpeg(self, test_data_dir: Path) -> None:
        """Test reading JPEG file."""
        jpeg_files = list(test_data_dir.glob("*.jpeg"))
        if not jpeg_files:
            pytest.skip("No JPEG test files available")

        reader = DataFile.get_reader(jpeg_files[0])
        data = reader.get_data()

        assert isinstance(data, np.ndarray)
        assert data.dtype in [np.uint8, np.float32, np.float64]

    def test_jpeg_data_shape(self, test_data_dir: Path) -> None:
        """Test JPEG data has expected shape."""
        jpeg_files = list(test_data_dir.glob("*.jpeg"))
        if not jpeg_files:
            pytest.skip("No JPEG test files available")

        reader = DataFile.get_reader(jpeg_files[0])
        data = reader.get_data()

        # 2D grayscale or 3D color
        assert data is not None
        assert data.ndim in [2, 3]


class TestDICOMReader:
    """Tests for DICOM reader."""

    def test_read_dicom(self, test_data_dir: Path) -> None:
        """Test reading DICOM file if available."""
        dcm_files = list(test_data_dir.glob("*.dcm")) + list(
            test_data_dir.glob("*.dicom")
        )
        if not dcm_files:
            pytest.skip("No DICOM test files available")

        reader = DataFile.get_reader(dcm_files[0])
        data = reader.get_data()

        assert isinstance(data, np.ndarray)


class TestNIfTIReader:
    """Tests for NIfTI reader."""

    def test_read_nifti(self, test_data_dir: Path) -> None:
        """Test reading NIfTI file if available."""
        nii_files = list(test_data_dir.glob("*.nii")) + list(
            test_data_dir.glob("*.nii.gz")
        )
        if not nii_files:
            pytest.skip("No NIfTI test files available")

        reader = DataFile.get_reader(nii_files[0])
        data = reader.get_data()

        assert isinstance(data, np.ndarray)

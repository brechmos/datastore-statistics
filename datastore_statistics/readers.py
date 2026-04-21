"""File readers for various medical image formats."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import imageio
import nibabel
import nibabel.nifti1
import numpy as np
import pydicom


class DataFile:
    """Base class for file readers.

    Defines the interface for reading different image formats and provides
    a factory method to instantiate the correct reader based on file extension.
    """

    _filename: Union[str, Path]

    def __init__(self) -> None:
        pass

    def get_data(self) -> Optional[np.ndarray]:
        """Return the image data as a numpy array."""
        return None

    def get_shape(self) -> Optional[Tuple[int, ...]]:
        """Return the shape of the image data."""
        return None

    def get_voxel_spacing(self) -> Optional[Tuple[float, ...]]:
        """Return voxel/pixel spacing as a tuple of floats (in mm where applicable).

        Returns None if spacing information is not available.
        """
        return None

    def get_type(self) -> Optional[str]:
        """Return the type of reader as a string."""
        return None

    def __str__(self) -> str:
        metrics = self.get_metrics()
        return (
            f'{self._filename}: min {metrics["min"]:0.1f}, '
            f'mu {metrics["mean"]:0.1f}, '
            f'med {metrics["median"]:0.1f} '
            f'max {metrics["max"]:0.1f}'
        )

    @staticmethod
    def get_reader(filename: Union[str, Path]) -> "DataFile":
        """Factory method to return the correct reader for a given filename.

        Args:
            filename: Path to the image file

        Returns:
            Appropriate reader instance (ImageIO, DICOM, or NII)

        Raises:
            ValueError: If file type is not supported
        """
        if isinstance(filename, str):
            filename = Path(filename)

        if filename.suffix == ".dcm":
            return DICOM(filename)
        elif ".nii" in str(filename).lower():
            return NII(filename)
        elif filename.suffix in [".jpeg", ".jpg", ".png", ".tiff", ".tif"]:
            return ImageIO(filename)
        else:
            raise ValueError(f"No reader available for file type: {filename.suffix}")

    def get_metrics(self) -> Dict[str, Any]:
        """Return basic statistics on the image data."""
        data = self.get_data()
        return {
            "mean": np.mean(data),
            "std": np.std(data),
            "median": np.median(data),
            "min": np.min(data),
            "max": np.max(data),
            "p0.1": np.percentile(data, 0.1),
            "p99.9": np.percentile(data, 99.9),
        }


class ImageIO(DataFile):
    """Reader for standard image formats (JPEG, PNG, TIFF) using imageio."""

    def __init__(self, filename: Union[str, Path]) -> None:
        super().__init__()
        self._filename = filename
        self._data: np.ndarray = np.asarray(imageio.imread(self._filename))

    def get_type(self) -> str:
        return "imageio"

    def get_data(self) -> np.ndarray:
        return self._data

    def get_shape(self) -> Tuple[int, ...]:
        return tuple(self._data.shape)


class DICOM(DataFile):
    """Reader for DICOM medical image files."""

    def __init__(self, filename: Union[str, Path]) -> None:
        super().__init__()
        self._filename = filename
        self._object = pydicom.dcmread(self._filename)

    def get_type(self) -> str:
        return "dicom"

    def get_data(self) -> np.ndarray:
        data: np.ndarray = np.asarray(self._object.pixel_array)
        return data

    def get_shape(self) -> Tuple[int, ...]:
        data: np.ndarray = np.asarray(self._object.pixel_array)
        return tuple(data.shape)

    def get_voxel_spacing(self) -> Optional[Tuple[float, ...]]:
        ds = self._object
        spacing: list[float] = []
        if hasattr(ds, "PixelSpacing") and ds.PixelSpacing:
            spacing = [float(s) for s in ds.PixelSpacing]
        if hasattr(ds, "SliceThickness") and ds.SliceThickness:
            spacing.append(float(ds.SliceThickness))
        return tuple(spacing) if spacing else None


class NII(DataFile):
    """Reader for NIfTI neuroimaging files."""

    def __init__(self, filename: Union[str, Path]) -> None:
        super().__init__()
        self._filename = filename
        self._nii: nibabel.nifti1.Nifti1Image = (
            nibabel.nifti1.Nifti1Image.from_filename(str(filename))
        )

    def get_type(self) -> str:
        return "nii"

    def get_data(self) -> np.ndarray:
        data: np.ndarray = np.asarray(self._nii.get_fdata())
        return data

    def get_shape(self) -> Tuple[int, ...]:
        return tuple(int(d) for d in self._nii.shape)

    def get_voxel_spacing(self) -> Optional[Tuple[float, ...]]:
        zooms = self._nii.header.get_zooms()
        return tuple(float(z) for z in zooms) if zooms else None

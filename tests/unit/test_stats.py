"""Unit tests for datastore_statistics.stats module."""

from pathlib import Path
from typing import Any, Dict, List

import nibabel as nib
import numpy as np
import pytest

from datastore_statistics.stats import (
    add_stats,
    compute_efc,
    compute_fber,
    compute_outliers,
    otsu_threshold,
)


class TestOtsuThreshold:
    """Tests for otsu_threshold function.

    Otsu's method finds the intensity threshold that minimizes the
    intra-class variance between two groups of pixels (foreground and
    background). It works best on images with a bimodal histogram and
    degrades gracefully to edge values for degenerate inputs (constant
    or all-zero images).
    """

    def test_bimodal_image(self, bimodal_image: np.ndarray) -> None:
        """Test Otsu threshold on bimodal image.

        A bimodal image has two distinct intensity peaks (background=0,
        foreground=200). Otsu should place the threshold between them,
        cleanly separating the two populations. The exact value depends
        on histogram bin edges, so we just check it falls in (0, 200).
        """
        thresh = otsu_threshold(bimodal_image)
        assert 0 < thresh < 200

    def test_constant_image(self, constant_image: np.ndarray) -> None:
        """Test Otsu threshold on constant image returns that value.

        When every pixel has the same intensity (128), there is no
        separation to find. Otsu defaults to that constant value since
        all histogram weight sits in a single bin.
        """
        thresh = otsu_threshold(constant_image)
        assert thresh == 128

    def test_blank_image(self, blank_image: np.ndarray) -> None:
        """Test Otsu threshold on blank image.

        An all-zero image is a special case of constant — all weight is
        in bin 0, so the threshold should be 0.
        """
        thresh = otsu_threshold(blank_image)
        assert thresh == 0

    def test_random_image(self, sample_2d_image: np.ndarray) -> None:
        """Test Otsu threshold returns valid value.

        For any uint8 image, the threshold must lie within the valid
        intensity range [0, 255] regardless of content.
        """
        thresh = otsu_threshold(sample_2d_image)
        assert 0 <= thresh <= 255


class TestComputeEfc:
    """Tests for compute_efc function.

    EFC (Entropy Focus Criterion) quantifies image ghosting and blurring
    by measuring the entropy of voxel intensities. The result is normalized
    to [0, 1], where values closer to 1 indicate a more uniform (blurry/
    ghosted) image and values closer to 0 indicate a sharply focused image
    with intensity concentrated in fewer voxels.

    EFC is undefined (NaN) when all voxels are zero, since normalization
    by total intensity would involve division by zero.
    """

    def test_efc_returns_normalized_value(self, sample_2d_image: np.ndarray) -> None:
        """Test EFC returns value between 0 and 1.

        Any valid image with non-zero content should produce a normalized
        entropy value within the [0, 1] range.
        """
        efc = compute_efc(sample_2d_image)
        assert 0 <= efc <= 1

    def test_efc_blank_image(self, blank_image: np.ndarray) -> None:
        """Test EFC on blank image returns NaN.

        An all-zero image has zero total intensity, making the entropy
        normalization undefined (division by zero), so we expect NaN.
        """
        efc = compute_efc(blank_image)
        assert np.isnan(efc)

    def test_efc_3d_volume(self, sample_3d_volume: np.ndarray) -> None:
        """Test EFC works on 3D volumes.

        EFC flattens the input before computing entropy, so it should
        handle 3D data the same way as 2D and still return a value in [0, 1].
        """
        efc = compute_efc(sample_3d_volume)
        assert 0 <= efc <= 1


class TestComputeFber:
    """Tests for compute_fber function.

    FBER (Foreground-to-Background Energy Ratio) measures how well the
    foreground signal stands out from the background. It is computed as:

        FBER = mean(foreground^2) / mean(background^2)

    A high FBER indicates strong foreground signal relative to the
    background (good image quality). The metric is undefined (NaN) when
    either the foreground or background region is empty, or when the
    background energy is zero (division by zero).
    """

    def test_fber_bimodal_image(self) -> None:
        """Test FBER on bimodal image returns high ratio.

        A clear two-level image (background=10, foreground=200) should
        produce a large FBER because the foreground energy (200^2=40000)
        vastly exceeds the background energy (10^2=100), giving ~400.
        """
        # Use float64 to avoid uint8 overflow when squaring
        img = np.ones((64, 64), dtype=np.float64) * 10  # background = 10
        img[16:48, 16:48] = 200  # foreground = 200
        mask = img > 100
        fber = compute_fber(img, mask)
        assert fber > 1

    def test_fber_no_foreground(self, blank_image: np.ndarray) -> None:
        """Test FBER with no foreground pixels returns NaN.

        When the mask selects zero pixels as foreground, the foreground
        mean energy is undefined, so FBER should be NaN.
        """
        mask = blank_image > 100
        fber = compute_fber(blank_image, mask)
        assert np.isnan(fber)

    def test_fber_all_foreground(self, constant_image: np.ndarray) -> None:
        """Test FBER with all foreground returns NaN.

        When every pixel is foreground, there are no background pixels,
        making the background energy undefined, so FBER should be NaN.
        """
        mask = constant_image > 0
        fber = compute_fber(constant_image, mask)
        assert np.isnan(fber)

    def test_fber_zero_background_energy(self) -> None:
        """Test FBER with zero background energy returns NaN.

        Background pixels are all zero, so mean(background^2) = 0.
        Division by zero makes FBER undefined, so we expect NaN.
        """
        img = np.zeros((64, 64))
        img[16:48, 16:48] = 100
        mask = img > 50
        fber = compute_fber(img, mask)
        assert np.isnan(fber)


class TestAddStats:
    """Tests for add_stats function."""

    def test_add_stats_basic(self, test_data_dir: Path) -> None:
        """Test add_stats populates expected fields."""
        jpeg_files = list(test_data_dir.glob("*.jpeg"))
        if not jpeg_files:
            pytest.skip("No JPEG test files available")

        file_info: Dict[str, Any] = {"filename": jpeg_files[0]}
        add_stats(file_info)

        expected_keys = [
            "data_shape",
            "foreground_pct",
            "min",
            "mean",
            "max",
            "std",
            "p25",
            "median",
            "p75",
            "efc",
            "fber",
        ]
        for key in expected_keys:
            assert key in file_info, f"Missing key: {key}"

    def test_add_stats_with_mask_threshold(self, test_data_dir: Path) -> None:
        """Test add_stats with manual mask threshold."""
        jpeg_files = list(test_data_dir.glob("*.jpeg"))
        if not jpeg_files:
            pytest.skip("No JPEG test files available")

        file_info: Dict[str, Any] = {"filename": jpeg_files[0]}
        add_stats(file_info, mask_threshold=50)

        assert "foreground_pct" in file_info
        # With threshold, foreground should be less than 100%
        # (depends on image content, but should be valid)
        assert 0 <= file_info["foreground_pct"] <= 100

    def test_add_stats_with_mask_auto(self, test_data_dir: Path) -> None:
        """Test add_stats with automatic Otsu masking."""
        jpeg_files = list(test_data_dir.glob("*.jpeg"))
        if not jpeg_files:
            pytest.skip("No JPEG test files available")

        file_info: Dict[str, Any] = {"filename": jpeg_files[0]}
        add_stats(file_info, mask_auto=True)

        assert "auto_threshold" in file_info
        assert "foreground_pct" in file_info


class TestComputeOutliers:
    """Tests for compute_outliers function."""

    def test_zscore_computation(self, sample_file_list: List[Dict[str, Any]]) -> None:
        """Test z-score computation adds zscore column."""
        result = compute_outliers(sample_file_list, "mean", "zscore")
        assert all("zscore-mean" in f for f in result)

    def test_zscore_filtering(self, sample_file_list: List[Dict[str, Any]]) -> None:
        """Test z-score filtering with threshold."""
        result = compute_outliers(
            sample_file_list,
            "mean",
            "zscore",
            threshold_value=2.0,
            threshold_direction="absolute",
        )
        # img4 has mean=200, which is an outlier
        assert len(result) < len(sample_file_list)

    def test_iqr_computation(self, sample_file_list: List[Dict[str, Any]]) -> None:
        """Test IQR computation adds iqr_scale column."""
        result = compute_outliers(sample_file_list, "mean", "iqr")
        assert all("iqr_scale-mean" in f for f in result)

    def test_no_threshold_returns_all(
        self, sample_file_list: List[Dict[str, Any]]
    ) -> None:
        """Test that no threshold returns all files."""
        result = compute_outliers(sample_file_list, "mean", "zscore")
        assert len(result) == len(sample_file_list)

    def test_threshold_direction_higher(
        self, sample_file_list: List[Dict[str, Any]]
    ) -> None:
        """Test threshold with 'higher' direction."""
        result = compute_outliers(
            sample_file_list,
            "mean",
            "zscore",
            threshold_value=1.5,
            threshold_direction="higher",
        )
        # Should only include positive outliers
        for f in result:
            assert f["zscore-mean"] > 1.5

    def test_threshold_direction_lower(
        self, sample_file_list: List[Dict[str, Any]]
    ) -> None:
        """Test threshold with 'lower' direction."""
        result = compute_outliers(
            sample_file_list,
            "mean",
            "zscore",
            threshold_value=1.5,
            threshold_direction="lower",
        )
        # Should only include negative outliers
        for f in result:
            assert f["zscore-mean"] < -1.5


class TestSpatialMetrics:
    """Tests for spatial metrics computed by add_stats."""

    def test_spatial_metrics_nifti(self, temp_dir: Path) -> None:
        """Test spatial metrics on a NIfTI volume."""
        data = np.random.randint(0, 256, size=(64, 64, 32), dtype=np.uint8)
        affine = np.diag([1.5, 1.5, 3.0, 1.0])
        img = nib.Nifti1Image(data, affine)
        path = temp_dir / "test.nii.gz"
        nib.save(img, str(path))

        file_info: Dict[str, Any] = {"filename": path}
        add_stats(file_info)

        assert file_info["n_slices"] == 32
        assert file_info["matrix_x"] == 64
        assert file_info["matrix_y"] == 64
        assert file_info["voxel_x"] == 1.5
        assert file_info["voxel_y"] == 1.5
        assert file_info["voxel_z"] == 3.0
        assert file_info["fov_x"] == 64 * 1.5
        assert file_info["fov_y"] == 64 * 1.5
        assert file_info["fov_z"] == 32 * 3.0

    def test_center_of_mass_nifti(self, temp_dir: Path) -> None:
        """Test center of mass on a volume with off-center foreground."""
        data = np.zeros((64, 64, 32), dtype=np.uint8)
        # Place bright block in upper-left region
        data[5:15, 5:15, 5:15] = 200
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        img = nib.Nifti1Image(data, affine)
        path = temp_dir / "test_com.nii.gz"
        nib.save(img, str(path))

        file_info: Dict[str, Any] = {"filename": path}
        add_stats(file_info, mask_auto=True)

        # Center of mass of block [5:15] = 9.5 in voxels → 9.5 * 2.0 = 19.0 mm
        assert file_info["com_x"] == 9.5 * 2.0
        assert file_info["com_y"] == 9.5 * 2.0
        assert file_info["com_z"] == 9.5 * 2.0

    def test_spatial_metrics_2d_image(self, test_data_dir: Path) -> None:
        """Test spatial metrics on a 2D image (no voxel spacing)."""
        jpeg_files = list(test_data_dir.glob("*.jpeg"))
        if not jpeg_files:
            pytest.skip("No JPEG test files available")

        file_info: Dict[str, Any] = {"filename": jpeg_files[0]}
        add_stats(file_info)

        assert file_info["n_slices"] == 1
        assert file_info["matrix_x"] > 0
        assert file_info["matrix_y"] > 0
        assert np.isnan(file_info["voxel_x"])
        assert np.isnan(file_info["fov_x"])

    def test_header_only_skips_pixel_metrics(self, temp_dir: Path) -> None:
        """Test that requesting only header metrics skips pixel data."""
        data = np.random.randint(0, 256, size=(64, 64, 32), dtype=np.uint8)
        affine = np.diag([1.5, 1.5, 3.0, 1.0])
        img = nib.Nifti1Image(data, affine)
        path = temp_dir / "test_header_only.nii.gz"
        nib.save(img, str(path))

        file_info: Dict[str, Any] = {"filename": path}
        add_stats(file_info, metrics={"n_slices", "voxel_x"})

        assert file_info["n_slices"] == 32
        assert file_info["voxel_x"] == 1.5
        # Pixel metrics should not be present
        assert "mean" not in file_info
        assert "efc" not in file_info
        assert "com_x" not in file_info

    def test_metrics_filter_pixel_only(self, temp_dir: Path) -> None:
        """Test requesting only pixel metrics skips spatial."""
        data = np.random.randint(0, 256, size=(64, 64, 32), dtype=np.uint8)
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        img = nib.Nifti1Image(data, affine)
        path = temp_dir / "test_pixel_only.nii.gz"
        nib.save(img, str(path))

        file_info: Dict[str, Any] = {"filename": path}
        add_stats(file_info, metrics={"mean", "efc"})

        assert "mean" in file_info
        assert "efc" in file_info
        assert "n_slices" not in file_info
        assert "fber" not in file_info

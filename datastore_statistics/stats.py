"""Statistics computation for image data."""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import ndimage

from .readers import DataFile


def otsu_threshold(data: np.ndarray) -> float:
    """Compute Otsu's threshold for separating foreground/background."""
    data_flat = data.flatten()

    # Handle edge case: constant image
    if np.min(data_flat) == np.max(data_flat):
        return float(np.min(data_flat))

    # Use 256 bins for histogram
    hist, bin_edges = np.histogram(data_flat, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute cumulative sums and means (avoid division by zero)
    weight1 = np.cumsum(hist).astype(float)
    weight2 = np.cumsum(hist[::-1])[::-1].astype(float)

    # Avoid division by zero
    weight1[weight1 == 0] = 1
    weight2[weight2 == 0] = 1

    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Compute between-class variance
    variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Find threshold that maximizes variance
    idx = np.argmax(variance)
    return float(bin_centers[idx])


def compute_efc(data: np.ndarray) -> float:
    """Compute Entropy Focus Criterion.

    Lower values indicate sharper/better focused images.
    Higher values indicate blur, ghosting, or motion artifacts.

    Args:
        data: Image data as numpy array

    Returns:
        EFC value (float)
    """
    data_flat: np.ndarray = data.flatten().astype(float)

    # Normalize to sum to 1
    data_sum: float = float(np.sum(data_flat))
    if data_sum == 0:
        return float(np.nan)

    data_norm = data_flat / data_sum

    # Remove zeros for log calculation
    data_norm = data_norm[data_norm > 0]

    # Compute entropy
    efc: float = float(-np.sum(data_norm * np.log(data_norm)))

    # Normalize by maximum possible entropy
    n = len(data_flat)
    efc_max = np.log(n) if n > 0 else 1
    efc_normalized = efc / efc_max if efc_max > 0 else 0

    return float(efc_normalized)


def compute_fber(data: np.ndarray, mask: np.ndarray) -> float:
    """Compute Foreground-Background Energy Ratio.

    Higher values indicate better signal quality (more foreground energy).
    Lower values indicate noisy/poor quality images.

    Args:
        data: Image data as numpy array
        mask: Boolean mask (True = foreground)

    Returns:
        FBER value (float)
    """
    if not np.any(mask) or np.all(mask):
        return float(np.nan)

    foreground = data[mask]
    background = data[~mask]

    # Energy = mean of squared values
    fg_energy = np.mean(foreground**2) if len(foreground) > 0 else 0
    bg_energy = np.mean(background**2) if len(background) > 0 else 0

    if bg_energy == 0:
        return float(np.nan)

    return float(fg_energy / bg_energy)


def _add_spatial_metrics(file: Dict[str, Any], reader: DataFile) -> None:
    """Compute spatial metrics from image header (no pixel data needed)."""
    shape = reader.get_shape()
    file["data_shape"] = shape
    file["n_slices"] = shape[2] if shape is not None and len(shape) >= 3 else 1
    file["matrix_x"] = shape[0] if shape is not None else 0
    file["matrix_y"] = shape[1] if shape is not None and len(shape) >= 2 else 1

    voxel_spacing = reader.get_voxel_spacing()
    if voxel_spacing is not None:
        file["voxel_x"] = voxel_spacing[0] if len(voxel_spacing) > 0 else np.nan
        file["voxel_y"] = voxel_spacing[1] if len(voxel_spacing) > 1 else np.nan
        file["voxel_z"] = voxel_spacing[2] if len(voxel_spacing) > 2 else np.nan
        file["fov_x"] = file["matrix_x"] * file["voxel_x"]
        file["fov_y"] = file["matrix_y"] * file["voxel_y"]
        if shape is not None and len(shape) >= 3 and len(voxel_spacing) > 2:
            file["fov_z"] = file["n_slices"] * file["voxel_z"]
        else:
            file["fov_z"] = np.nan
    else:
        file["voxel_x"] = file["voxel_y"] = file["voxel_z"] = np.nan
        file["fov_x"] = file["fov_y"] = file["fov_z"] = np.nan


def _add_intensity_metrics(file: Dict[str, Any], masked_data: np.ndarray) -> None:
    """Compute intensity statistics on masked data."""
    if masked_data.size > 0:
        file["min"] = np.min(masked_data)
        file["mean"] = np.mean(masked_data)
        file["max"] = np.max(masked_data)
        file["std"] = np.std(masked_data)
        pct_arr: np.ndarray = np.asarray(np.percentile(masked_data, [25, 50, 75]))
        file["p25"] = pct_arr[0]
        file["median"] = pct_arr[1]
        file["p75"] = pct_arr[2]
    else:
        file["min"] = file["mean"] = file["max"] = file["std"] = np.nan
        file["p25"] = file["median"] = file["p75"] = np.nan


def _add_com_metrics(
    file: Dict[str, Any],
    mask: np.ndarray,
    voxel_spacing: Optional[Tuple[float, ...]],
) -> None:
    """Compute center of mass metrics, in mm if spacing is available."""
    if np.any(mask):
        com_voxel = ndimage.center_of_mass(mask.astype(float))
        if voxel_spacing is not None:
            for i, axis in enumerate(["x", "y", "z"]):
                if i < len(com_voxel) and i < len(voxel_spacing):
                    file[f"com_{axis}"] = com_voxel[i] * voxel_spacing[i]
                else:
                    file[f"com_{axis}"] = np.nan
        else:
            file["com_x"] = com_voxel[0] if len(com_voxel) > 0 else np.nan
            file["com_y"] = com_voxel[1] if len(com_voxel) > 1 else np.nan
            file["com_z"] = com_voxel[2] if len(com_voxel) > 2 else np.nan
    else:
        file["com_x"] = file["com_y"] = file["com_z"] = np.nan


def _add_efc_metric(file: Dict[str, Any], data: np.ndarray) -> None:
    """Compute Entropy Focus Criterion metric."""
    file["efc"] = compute_efc(data)


def _add_fber_metric(file: Dict[str, Any], data: np.ndarray, mask: np.ndarray) -> None:
    """Compute Foreground-Background Energy Ratio metric."""
    file["fber"] = compute_fber(data, mask)


def add_stats(
    file: Dict[str, Any],
    mask_threshold: Optional[float] = None,
    mask_auto: bool = False,
    metrics: Optional[Set[str]] = None,
) -> None:
    """Compute statistics for an image file.

    Only computes the metrics specified. If metrics is None, computes all.

    Args:
        file: Dict with 'filename' key, will be updated with stats
        mask_threshold: Ignore pixels below this value (optional)
        mask_auto: Use Otsu's method to auto-determine threshold
        metrics: Set of metric names to compute. None means all.
    """
    reader = DataFile.get_reader(file["filename"])

    # Determine which groups of metrics are needed
    need_spatial = metrics is None or bool(
        metrics
        & {
            "n_slices",
            "matrix_x",
            "matrix_y",
            "voxel_x",
            "voxel_y",
            "voxel_z",
            "fov_x",
            "fov_y",
            "fov_z",
        }
    )
    need_intensity = metrics is None or bool(
        metrics
        & {
            "min",
            "mean",
            "max",
            "std",
            "p25",
            "median",
            "p75",
            "foreground_pct",
        }
    )
    need_com = metrics is None or bool(metrics & {"com_x", "com_y", "com_z"})
    need_efc = metrics is None or "efc" in metrics
    need_fber = metrics is None or "fber" in metrics
    need_pixel_data = need_intensity or need_com or need_efc or need_fber

    if need_spatial:
        _add_spatial_metrics(file, reader)

    if not need_pixel_data:
        return

    data = reader.get_data()
    if data is None:
        return
    if not need_spatial:
        file["data_shape"] = data.shape

    # Compute Otsu threshold once (needed for masking, COM, and FBER)
    auto_thresh = otsu_threshold(data)

    # Determine mask and masked data.
    # Three modes control which voxels are treated as "foreground":
    #   1. mask_auto  – use the automatically computed threshold (e.g. Otsu)
    #   2. mask_threshold – use an explicit, user-supplied threshold
    #   3. neither – no masking; all voxels are included as foreground
    if mask_auto:
        # Record the auto threshold and keep only voxels above it
        file["auto_threshold"] = auto_thresh
        mask = data > auto_thresh
        masked_data = data[mask]
        file["foreground_pct"] = 100.0 * np.sum(mask) / data.size
    elif mask_threshold is not None:
        # Use the user-provided threshold to separate foreground from background
        mask = data > mask_threshold
        masked_data = data[mask]
        file["foreground_pct"] = 100.0 * np.sum(mask) / data.size
    else:
        # No masking requested – compute a mask for potential later use,
        # but treat all voxels as foreground (100%)
        mask = data > auto_thresh
        masked_data = data.flatten()
        file["foreground_pct"] = 100.0

    if need_intensity:
        _add_intensity_metrics(file, masked_data)
    if need_com:
        _add_com_metrics(file, mask, reader.get_voxel_spacing())
    if need_efc:
        _add_efc_metric(file, data)
    if need_fber:
        _add_fber_metric(file, data, mask)


def compute_outliers(
    file_list: List[Dict[str, Any]],
    metric: str,
    statistic: str,
    threshold_value: Optional[float] = None,
    threshold_direction: str = "absolute",
) -> List[Dict[str, Any]]:
    """Compute outlier statistics and apply threshold filtering.

    Args:
        file_list: List of dicts with computed stats
        metric: Which metric to analyze (mean, median, etc.)
        statistic: Method for outlier detection (zscore, iqr, stats)
        threshold_value: Threshold for filtering (optional)
        threshold_direction: Direction for threshold (absolute, lower, higher)

    Returns:
        Filtered list of files (or all files if no threshold)
    """
    from .utils import threshold as apply_threshold

    if statistic == "zscore":
        values = np.array([x[metric] for x in file_list])
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        zscores = (
            (values - mean_val) / std_val if std_val > 0 else np.zeros_like(values)
        )

        for zscore, file in zip(zscores, file_list):
            file["zscore-" + metric] = zscore

        if threshold_value is not None:
            file_list = [
                f
                for f in file_list
                if apply_threshold(
                    f["zscore-" + metric], threshold_value, threshold_direction
                )
            ]

    elif statistic == "iqr":
        values = np.array([x[metric] for x in file_list])
        iqr = float(np.diff(np.nanpercentile(values, [25, 75]))[0])

        for file in file_list:
            file["iqr_scale-" + metric] = file[metric] / iqr if iqr > 0 else np.nan

        if threshold_value is not None:
            file_list = [
                f
                for f in file_list
                if apply_threshold(
                    f["iqr_scale-" + metric], threshold_value, threshold_direction
                )
            ]

    return file_list

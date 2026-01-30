# Datastore Statistics

Lightweight outlier detection for medical image datasets. Quickly scan thousands of images to find the ones that don't belong - corrupted files, wrong acquisitions, or quality issues.

## Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -e .

# For HTML report with thumbnails
pip install -e ".[report]"

# For development (includes pre-commit hooks)
pip install -e ".[dev]"
pre-commit install
```

## Important: Homogeneous Data

This tool computes group statistics (mean, standard deviation) across all images in the directory and flags outliers relative to the group. For meaningful results, **all images should be of the same type/modality** (e.g., all T1-weighted, all rsfMRI, etc.). Mixing different scan types will produce unreliable z-scores since the group statistics won't represent any single acquisition type. Use `--filename-regexp` to filter to a specific scan type if your directory contains mixed data.

## Quick Start

```bash
# Find images with unusual mean intensity (z-score > 2)
ds-stats --data-directory /path/to/images --metric mean --threshold 2

# Check multiple metrics simultaneously
ds-stats --data-directory /path/to/images --metric mean median efc --threshold 2

# Generate visual HTML report
ds-stats --data-directory /path/to/images --metric mean --threshold 2 --html-report report.html

# Use automatic background masking (Otsu's method)
ds-stats --data-directory /path/to/images --metric mean --mask-auto --threshold 2
```

## Features

### Supported Formats
- **DICOM** (.dcm, .dicom)
- **NIfTI** (.nii, .nii.gz)
- **Standard images** (.jpg, .jpeg, .png)

### 3D Volume Handling
- Statistics are computed over the **entire volume**, not per-slice
- HTML report thumbnails display the **middle slice** for 3D data

### Outlier Detection Methods
- **Z-score**: Flag images where a metric deviates more than N standard deviations from the mean
- **IQR**: Flag images outside the interquartile range (robust to extreme outliers)

### Metrics

You can analyze **one or multiple metrics** simultaneously. When multiple metrics are specified, images are flagged if they're outliers on **any** of the metrics.

| Metric | Description |
|--------|-------------|
| `mean`, `median`, `min`, `max`, `std` | Basic intensity statistics |
| `p25`, `p75` | Percentiles |
| `efc` | Entropy Focus Criterion - detects blur, ghosting, motion artifacts (higher = worse) |
| `fber` | Foreground-Background Energy Ratio - measures signal quality (higher = better) |
| `n_slices` | Number of slices (for 3D volumes) |
| `com_x`, `com_y`, `com_z` | Center of mass coordinates |

### Background Masking
- `--mask-threshold VALUE`: Ignore pixels below this intensity
- `--mask-auto`: Automatically determine threshold using Otsu's method

### Output Formats
- `screen`: Pretty-printed table (default)
- `csv`: Comma-separated values
- `json`: JSON format
- `--html-report PATH`: Visual report with image thumbnails

## Examples

```bash
# Check median intensity, flag values more than 1.5 IQR above normal
ds-stats --data-directory data/ --metric median --statistic iqr --threshold 1.5 --threshold-direction higher

# Check multiple metrics - flags images that are outliers on ANY metric
ds-stats --data-directory data/ --metric mean efc fber --threshold 2

# Export to CSV for further analysis
ds-stats --data-directory data/ --metric mean median --output-format csv > results.csv

# Only process NIfTI files
ds-stats --data-directory data/ --filename-regexp ".*\.nii(\.gz)?"

# Check volume geometry metrics
ds-stats --data-directory data/ --metric n_slices com_x com_y com_z --threshold 3

# Save log to file
ds-stats --data-directory data/ --metric mean --threshold 2 --logfile results.log
```

## Grouping by Sidecar Metadata

When working with multi-site or heterogeneous datasets, raw intensity metrics will flag scanner differences rather than true quality issues. The `--group-by-sidecar` option lets you compute z-scores **within subgroups** defined by JSON sidecar fields, so you're comparing like with like.

This requires JSON sidecar files alongside your images (e.g., `image.nii.gz` with `image.json`), as produced by tools like [RADIFOX](https://github.com/jh-mipc/radifox) or [BIDS](https://bids-specification.readthedocs.io/).

```bash
# Group by a single field (e.g., field strength)
ds-stats --data-directory /path/to/images \
  --filename-regexp '.*T1-.*\.nii\.gz' \
  --metric mean std efc \
  --group-by-sidecar SeriesInfo.MagneticFieldStrength \
  --threshold 2 --mask-auto

# Group by multiple fields for tighter subgroups
ds-stats --data-directory /path/to/images \
  --filename-regexp '.*T1-.*\.nii\.gz' \
  --metric mean std efc \
  --group-by-sidecar SeriesInfo.MagneticFieldStrength \
                     SeriesInfo.EchoTrainLength \
                     SeriesInfo.AcquisitionDimension \
                     SeriesInfo.NumberOfAverages \
                     SeriesInfo.SeriesDescription \
                     SeriesInfo.SliceThickness \
  --threshold 2 --mask-auto --html-report report.html

# For multi-site data with intact DICOM headers, add vendor/site fields
ds-stats --data-directory /path/to/images \
  --filename-regexp '.*T1-.*\.nii\.gz' \
  --metric mean std efc \
  --group-by-sidecar SeriesInfo.MagneticFieldStrength \
                     SeriesInfo.Manufacturer \
                     SeriesInfo.ScannerModelName \
                     SeriesInfo.SeriesDescription \
                     SeriesInfo.SliceThickness \
  --threshold 2 --mask-auto --html-report report.html
```

**Why group by multiple fields?** MRI intensity values are arbitrary — two scanners produce different scales for the same tissue. Grouping by field strength alone still mixes 2D spin echo with 3D gradient echo acquisitions, which have fundamentally different intensity profiles. Adding `EchoTrainLength` separates SE from FSE, and `AcquisitionDimension` separates 2D from 3D. The result is subgroups where outliers reflect genuine quality issues rather than protocol differences.

| Sidecar field | What it separates |
|---|---|
| `MagneticFieldStrength` | 1.5T vs 3T scanners |
| `EchoTrainLength` | Spin Echo (1) vs Fast Spin Echo (>1) |
| `AcquisitionDimension` | 2D vs 3D acquisitions |
| `SliceThickness` | Different slice thickness protocols |
| `NumberOfAverages` | NEX/NSA — affects SNR and noise floor |
| `Manufacturer` | Scanner vendor (e.g., GE vs Siemens vs Philips) |
| `ScannerModelName` | Scanner model (e.g., Prisma vs Skyra) |
| `InstitutionName` | Per-site grouping for multi-site studies |

Fields use dot notation to traverse nested JSON (e.g., `SeriesInfo.MagneticFieldStrength`). Missing sidecars or fields are grouped as `_NO_SIDECAR` or `_NO_VALUE`. Groups with fewer than 3 images produce a warning since z-scores become unreliable.

## CLI Reference

```
ds-stats --help

Options:
  --data-directory PATH      Directory containing images (default: data/)
  --filename-regexp PATTERN  Regex to filter filenames
  --metric METRIC [METRIC...]  Metric(s) to analyze: mean, median, min, max, std, efc, fber, n_slices, com_x, com_y, com_z
                              Can specify multiple (e.g., --metric mean efc fber)
                              Flags images that are outliers on ANY specified metric
  --statistic METHOD         Detection method: zscore or iqr
  --threshold VALUE          Threshold for flagging outliers
  --threshold-direction DIR  absolute, lower, or higher
  --mask-threshold VALUE     Ignore pixels below this value
  --mask-auto                Auto-detect background with Otsu's method
  --output-format FORMAT     screen, csv, or json
  --html-report PATH         Generate HTML report with thumbnails
  --group-by-sidecar PATH [PATH...]  Group by JSON sidecar field(s) using dot notation
  --logfile PATH             Write output to log file
```

## To Do

- [ ] Load images from cloud storage (Azure Blob, S3)
- [ ] Reference mode: compute stats on one dataset, detect outliers in another

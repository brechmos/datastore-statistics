"""HTML report generation for datastore statistics."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# ---------------------------
# Image helpers
# ---------------------------


def image_to_base64(data: np.ndarray, max_size: int = 128) -> Optional[str]:
    """Convert image data to base64 encoded PNG for embedding in HTML."""
    try:
        from PIL import Image
    except ImportError:
        return None

    # Reduce to 2D by taking the middle slice along trailing dimensions
    while data.ndim > 2:
        data = data[..., data.shape[-1] // 2]

    data = data.astype(float)
    if data.max() > data.min():
        data = (data - data.min()) / (data.max() - data.min()) * 255
    data = data.astype(np.uint8)

    img = Image.fromarray(data)

    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        img = img.resize(
            (int(img.size[0] * scale), int(img.size[1] * scale)),
            Image.Resampling.LANCZOS,
        )

    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------
# Metadata filtering helpers
# ---------------------------

META_IGNORE_PREFIXES = (
    "InputHash",
    "NiftiHash",
    "SourceHash",
    "SeriesUID",
    "StudyUID",
    "SourcePath",
    "NiftiName",
    "LookupName",
    "ManualName",
    "PredictedName",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "AcqDateTime",
    "DeviceIdentifier",
    "ConversionSoftwareVersions",
    "__version__",
)

# Only show fields that help explain imaging differences
META_WHITELIST = {
    # Spatial/geometry
    "SeriesInfo.AcquisitionDimension",
    "SeriesInfo.AcquiredResolution",
    "SeriesInfo.ReconResolution",
    "SeriesInfo.FieldOfView",
    "SeriesInfo.SliceThickness",
    "SeriesInfo.SliceSpacing",
    # Scanner/hardware
    "SeriesInfo.MagneticFieldStrength",  # MRI
    "SeriesInfo.Manufacturer",
    "SeriesInfo.ReceiveCoilName",  # MRI
    # Sequence identification
    "SeriesInfo.ImageType",
    "SeriesInfo.SeriesDescription",
    # Timing parameters (MRI)
    "SeriesInfo.EchoTime",  # MRI
    "SeriesInfo.EchoTrainLength",  # MRI
    "SeriesInfo.InversionTime",  # MRI
    "SeriesInfo.RepetitionTime",  # MRI
    # Acquisition parameters (MRI)
    "SeriesInfo.FlipAngle",  # MRI
    "SeriesInfo.PercentSampling",  # MRI
    "SeriesInfo.NumberOfAverages",  # MRI
    "SeriesInfo.PixelBandwidth",  # MRI
    "SeriesInfo.ScanOptions",  # MRI
    "SeriesInfo.SequenceVariant",  # MRI
    # CT parameters
    "SeriesInfo.ConvolutionKernel",  # CT - Reconstruction kernel (e.g., STANDARD, BONE)
    "SeriesInfo.KVP",  # CT - X-ray tube voltage in kVp (e.g., 80, 120)
    "SeriesInfo.XRayTubeCurrent",  # CT - Tube current in mA
    "SeriesInfo.ExposureTime",  # CT - Exposure duration in ms
    "SeriesInfo.Exposure",  # CT - Radiation exposure in mAs
    "SeriesInfo.ExposureInuAs",  # CT - Exposure in microampere-seconds
    "SeriesInfo.FilterType",  # CT - X-ray filter type
    "SeriesInfo.ExposureModulationType",  # CT - Dose modulation type
    "SeriesInfo.MultienergyCTAcquisition",  # CT - Multi-energy CT acquisition (YES/NO)
}


def filter_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact, diagnostically useful metadata."""
    filtered = {}

    for key, val in meta.items():
        if any(key.startswith(p) for p in META_IGNORE_PREFIXES):
            continue
        if key in META_WHITELIST:
            filtered[key.split(".")[-1]] = val

    return filtered


def differing_keys(dicts: List[Dict[str, Any]]) -> List[str]:
    """Return keys whose values differ across dicts."""
    keys = set().union(*(d.keys() for d in dicts))
    out = []
    for k in keys:
        vals = {str(d.get(k)) for d in dicts}
        if len(vals) > 1:
            out.append(k)
    return sorted(out)


# ---------------------------
# HTML report
# ---------------------------


def generate_html_report(
    file_list: List[Dict[str, Any]],
    output_path: Union[str, Path],
    metric: str = "mean",
    metrics: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    group_key: Optional[str] = None,
    group_totals: Optional[Dict[str, int]] = None,
    all_files: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Generate an HTML report with image thumbnails and statistics."""
    from .readers import DataFile
    from .sidecar import load_all_sidecar_metadata

    metrics = metrics or [metric]

    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Datastore Statistics Report</title>",
        "<style>",
        "body{font-family:Arial;margin:20px;background:#f5f5f5}",
        ".grid{display:grid;grid-template-columns:"
        "repeat(auto-fill,minmax(220px,1fr));gap:15px}",
        ".card{background:#fff;padding:10px;border-radius:8px;text-align:center}",
        ".card img{display:block;margin:0 auto}",
        ".no-image{background:#ddd;height:100px;"
        "display:flex;align-items:center;justify-content:center}",
        ".compare{background:#fff;margin-top:15px;padding:10px;border-radius:8px}",
        ".path-row{font-family:monospace;font-size:12px;padding:4px}",
        ".outlier-path{background:#fde8e8}",
        ".normal-path{background:#e8fde8}",
        ".meta{font-size:11px;color:#444;margin-left:12px}",
        ".stats{font-size:11px;color:#333;margin-top:6px}",
        ".stats td{padding:1px 6px}",
        ".flagged{color:#c00;font-weight:bold}",
        "</style></head><body>",
        "<h1>Datastore Statistics Report</h1>",
    ]

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for f in file_list:
        g = str(f.get(group_key, "_all")) if group_key else "_all"
        groups.setdefault(g, []).append(f)

    all_by_group: Dict[str, List[Dict[str, Any]]] = {}
    if all_files:
        if group_key:
            for f in all_files:
                all_by_group.setdefault(str(f.get(group_key)), []).append(f)
        else:
            all_by_group["_all"] = list(all_files)

    for group, files in groups.items():
        html.append(f"<h2>Group: {group}</h2>")
        html.append("<div class='grid'>")

        for f in files:
            html.append("<div class='card'>")
            try:
                data = DataFile.get_reader(f["filename"]).get_data()
                img = image_to_base64(data)
                html.append(
                    f"<img src='data:image/png;base64,{img}'>"
                    if img
                    else "<div class='no-image'>No preview</div>"
                )
            except Exception as exc:
                html.append(f"<div class='no-image'>Error: {exc}</div>")

            html.append(f"<div>{Path(f['filename']).name}</div>")

            # Show metric values and scores
            html.append("<table class='stats'>")
            for m in metrics:
                val = f.get(m)
                # Find the score key (zscore- or iqr_scale-)
                score = None
                score_key = None
                for prefix in ("zscore-", "iqr_scale-"):
                    if prefix + m in f:
                        score = f[prefix + m]
                        score_key = prefix.rstrip("-")
                        break
                flagged = (
                    score is not None
                    and threshold is not None
                    and abs(float(score)) >= threshold
                )
                cls = " class='flagged'" if flagged else ""
                val_str = f"{val:.3g}" if isinstance(val, float) else str(val)
                score_str = f" ({score_key}: {score:.2f})" if score is not None else ""
                html.append(
                    f"<tr{cls}><td>{m}</td>" f"<td>{val_str}{score_str}</td></tr>"
                )
            html.append("</table>")

            html.append("</div>")

        html.append("</div>")

        # -------- comparison section --------
        if threshold and group in all_by_group:
            outliers = files
            normals = [
                f
                for f in all_by_group[group]
                if f["filename"] not in {o["filename"] for o in outliers}
            ]

            if normals:
                out_meta = [
                    filter_metadata(load_all_sidecar_metadata(f["filename"]) or {})
                    for f in outliers
                ]
                norm_meta = [
                    filter_metadata(load_all_sidecar_metadata(f["filename"]) or {})
                    for f in normals
                ]

                diff_keys = differing_keys(out_meta + norm_meta)

                html.append("<div class='compare'>")
                for f, meta in zip(outliers, out_meta):
                    path = f["filename"]
                    html.append(
                        f"<div class='path-row outlier-path'>OUTLIER: {path}</div>"
                    )
                    for k in diff_keys:
                        html.append(f"<div class='meta'>{k}: {meta.get(k)}</div>")

                html.append(
                    "<div style='margin:6px 0'>Non-outliers from same group:</div>"
                )
                for f, meta in zip(normals, norm_meta):
                    path = f["filename"]
                    html.append(
                        f"<div class='path-row normal-path'>NORMAL: {path}</div>"
                    )
                    for k in diff_keys:
                        html.append(f"<div class='meta'>{k}: {meta.get(k)}</div>")

                html.append("</div>")

    html.append("</body></html>")

    output_path = Path(output_path)
    output_path.write_text("\n".join(html))
    return output_path

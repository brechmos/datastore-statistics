import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


def resolve_sidecar(image_path: Path) -> Optional[Path]:
    """Find a JSON sidecar file for the given image path.

    Looks for a .json file with the same base name next to the image.
    Handles .nii.gz (double extension) and .nii, .dcm, .dicom extensions.
    """
    image_path = Path(image_path)
    if image_path.name.endswith(".nii.gz"):
        sidecar = image_path.with_name(image_path.name[: -len(".nii.gz")] + ".json")
    else:
        sidecar = image_path.with_suffix(".json")

    if sidecar.exists():
        return sidecar
    return None


def extract_json_path(sidecar_path: Path, json_path: str) -> Any:
    """Extract a value from a JSON file using dot-notation path.

    For example, 'SeriesInfo.MagneticFieldStrength' traverses:
    sidecar['SeriesInfo']['MagneticFieldStrength']

    Returns None if any key is missing.
    """
    try:
        data = json.loads(sidecar_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to read sidecar %s: %s", sidecar_path, exc)
        return None

    keys = json_path.split(".")
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return None
    return data


def get_group_value(image_path: Path, json_path: str) -> str:
    """Get the group label for an image from its sidecar.

    Returns a string suitable for use as a group key.
    Numeric values are rounded to 1 decimal place.
    """
    sidecar = resolve_sidecar(image_path)
    if sidecar is None:
        log.warning("No sidecar found for %s", image_path)
        return "_NO_SIDECAR"

    val = extract_json_path(sidecar, json_path)
    if val is None:
        log.warning("Field '%s' not found in sidecar for %s", json_path, image_path)
        return "_NO_VALUE"

    if isinstance(val, float):
        val = round(val, 1)

    return str(val)


def load_all_sidecar_metadata(image_path: Path) -> Optional[Dict[str, Any]]:
    """Load all metadata from a sidecar JSON file.

    Returns a flattened dictionary with dot-notation keys,
    or None if no sidecar exists.
    """
    sidecar = resolve_sidecar(image_path)
    if sidecar is None:
        return None

    try:
        data = json.loads(sidecar.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to read sidecar %s: %s", sidecar, exc)
        return None

    return _flatten_dict(data)


def _flatten_dict(d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    """Flatten a nested dictionary using dot notation for keys."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)

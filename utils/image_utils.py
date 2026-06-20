"""
utils/image_utils.py
Local, dependency-light (PIL-only) image loading and quality heuristics.

These heuristics are a *pre-filter / sanity backstop* only. The primary
visual judgment (does the image show the claimed damage, is it blurry,
is it the wrong object, etc.) is made by the Gemini vision call in
agents/vision_analyzer.py. Local heuristics never override the model --
they are unioned with the model's own quality flags so a single failure
mode (e.g. the model missing an obviously corrupt file) can't silently
pass through.
"""

from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageStat, ImageFilter, UnidentifiedImageError

import config
from utils.logger import get_logger

logger = get_logger(__name__)


def resolve_image_path(raw_path: str, base_dir: Path = config.DATASET_DIR) -> Path:
    """Resolve a path from claims.csv relative to the dataset directory if needed."""
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p
    candidate = base_dir / raw_path
    if candidate.exists():
        return candidate
    # Last resort: return as given so downstream loading fails loudly and
    # is reported, rather than silently swapped for something else.
    return p


def make_image_id(user_id: str, index: int) -> str:
    return f"{user_id}_img{index + 1}"


def load_image(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        logger.warning("Failed to load image %s: %s", path, exc)
        return None


def _blur_edge_variance(gray_image: Image.Image) -> float:
    """Cheap blur proxy: variance of an edge-detected grayscale image."""
    edges = gray_image.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(edges)
    return float(stat.stddev[0] ** 2)


def assess_image_quality(path: Path) -> Dict:
    """
    Returns a dict describing local quality signals for one image:
      {
        is_loadable: bool,
        width, height: int,
        brightness: float,
        blur_score: float,
        local_quality_flags: [subset of config.RISK_FLAGS],
      }
    """
    result = {
        "is_loadable": False,
        "width": 0,
        "height": 0,
        "brightness": 0.0,
        "blur_score": 0.0,
        "local_quality_flags": [],
    }

    img = load_image(path)
    if img is None:
        return result

    result["is_loadable"] = True
    result["width"], result["height"] = img.size

    gray = img.convert("L")
    brightness = float(ImageStat.Stat(gray).mean[0])
    blur_score = _blur_edge_variance(gray)

    result["brightness"] = brightness
    result["blur_score"] = blur_score

    flags: List[str] = []
    if (
        brightness < config.LOW_LIGHT_BRIGHTNESS_THRESHOLD
        or brightness > config.OVEREXPOSED_BRIGHTNESS_THRESHOLD
    ):
        flags.append("low_light_or_glare")

    if blur_score < config.BLUR_EDGE_VARIANCE_THRESHOLD:
        flags.append("blurry_image")

    if min(result["width"], result["height"]) < config.MIN_IMAGE_DIMENSION_PX:
        flags.append("cropped_or_obstructed")

    result["local_quality_flags"] = flags
    return result


def load_image_for_model(path: Path) -> Optional[Image.Image]:
    """
    Loads and lightly downsizes an image so it is safe to send to the
    Gemini API (keeps payload size and token usage bounded).
    """
    img = load_image(path)
    if img is None:
        return None

    max_dim = 1536
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    return img

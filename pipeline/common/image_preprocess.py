#!/usr/bin/env python3
"""Image preprocessing utilities: rotation detection, resizing, normalization."""

import logging
from typing import Tuple, Dict, Any, Optional

from PIL import Image

# Optional pytesseract for rotation detection
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    pytesseract = None


def detect_page_rotation(img: Image.Image, logger: logging.Logger) -> int:
    """Detect page rotation using Tesseract OSD; returns 0, 90, 180, or 270 degrees."""
    if not PYTESSERACT_AVAILABLE:
        logger.debug("pytesseract not available; skipping rotation detection")
        return 0

    try:
        osd_data = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        logger.debug("[ROTATION] Tesseract OSD data: %s", osd_data)
        rotation = osd_data.get("rotate", 0)
        confidence = osd_data.get("orientation_conf", 0.0)
        logger.debug("[ROTATION] Detected rotation=%d° (confidence=%.1f%%)", rotation, confidence)

        if confidence < 5.0:
            logger.debug("[ROTATION] Low confidence (%.1f%%), skipping rotation", confidence)
            return 0

        return int(rotation)
    except Exception as exc:
        logger.debug("[ROTATION] Detection failed: %s: %s", type(exc).__name__, exc)
        return 0


def resize_image_if_needed(
    img: Image.Image,
    max_size: Optional[int],
    logger: logging.Logger
) -> Tuple[Image.Image, bool]:
    """Resize image if longer edge exceeds max_size, maintaining aspect ratio."""
    if max_size is None:
        return img, False

    width, height = img.size
    longer_edge = max(width, height)

    if longer_edge <= max_size:
        return img, False

    scale = max_size / longer_edge
    new_width = int(width * scale)
    new_height = int(height * scale)

    logger.info(
        "[RESIZE] %dx%d → %dx%d (max=%dpx, scale=%.3f)",
        width, height, new_width, new_height, max_size, scale
    )

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS), True


def normalize_image_rgb(img: Image.Image) -> Image.Image:
    """Ensure image is in RGB mode."""
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def preprocess_image(
    img: Image.Image,
    max_input_size: Optional[int],
    logger: logging.Logger
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Preprocess image: detect rotation, rotate if needed, resize, normalize to RGB; returns (img, metadata)."""
    original_size = img.size
    meta: Dict[str, Any] = {
        "original_size": original_size,
        "detected_rotation": 0,
        "was_rotated": False,
        "was_resized": False,
    }

    # Ensure RGB for rotation detection
    img = normalize_image_rgb(img)

    # Detect and apply rotation
    detected_rotation = detect_page_rotation(img, logger)
    meta["detected_rotation"] = detected_rotation

    if detected_rotation > 0:
        logger.info(
            "[PREPROCESS] Rotation correction: detected=%d° → rotating image",
            detected_rotation
        )
        img = img.rotate(-detected_rotation, expand=True, resample=Image.Resampling.BICUBIC)
        meta["was_rotated"] = True
        logger.info(
            "[PREPROCESS] After rotation: %dx%d", img.size[0], img.size[1]
        )

    # Resize if needed
    img, was_resized = resize_image_if_needed(img, max_input_size, logger)
    meta["was_resized"] = was_resized

    meta["final_size"] = img.size

    return img, meta






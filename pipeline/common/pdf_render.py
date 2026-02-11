#!/usr/bin/env python3
"""PDF rendering utilities for OCR pipelines."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz
from PIL import Image

from lsextractor.pipeline.common.image_preprocess import (
    detect_page_rotation,
    resize_image_if_needed,
)


def get_pdf_page_count(pdf_path: Path | str) -> int:
    """Open PDF and return total page count."""
    try:
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF {pdf_path}: {type(exc).__name__}: {exc}") from exc


def render_one_pdf_page(
    pdf_path: Path | str,
    page_index: int,
    dpi: int,
    out_dir: Path | str,
    logger: logging.Logger,
    max_input_size: Optional[int] = None,
    apply_rotation_detection: bool = True,
) -> Tuple[Path, List[int], Dict[str, Any]]:
    """Render a single PDF page to PNG with rotation detection and resizing."""
    t0 = time.time()

    doc = fitz.open(pdf_path)
    try:
        if page_index < 0 or page_index >= len(doc):
            raise ValueError(f"Page index {page_index} out of range [0, {len(doc)-1}]")

        page = doc[page_index]

        try:
            page_rotation = int(page.rotation)
        except Exception:
            page_rotation = None

        page_rect = page.rect
        pdf_width_pt = page_rect.width
        pdf_height_pt = page_rect.height

        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        mode = "RGB" if pixmap.n < 4 else "RGBA"
        img = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
        if img.mode != "RGB":
            img = img.convert("RGB")

        rendered_size = [pixmap.width, pixmap.height]

        detected_rotation = 0
        was_rotated = False
        if apply_rotation_detection:
            detected_rotation = detect_page_rotation(img, logger)
            if detected_rotation > 0:
                img = img.rotate(-detected_rotation, expand=True, resample=Image.Resampling.BICUBIC)
                was_rotated = True
                logger.info(
                    "[RENDER][P%03d] Rotation correction: detected=%d° → rotated %d° (new size: %dx%d)",
                    page_index + 1, detected_rotation, -detected_rotation, img.size[0], img.size[1]
                )

        was_resized = False
        if max_input_size:
            img, was_resized = resize_image_if_needed(img, max_input_size, logger)
            if was_resized:
                logger.info(
                    "[RENDER][P%03d] Resized to %dx%d (max_input_size=%d)",
                    page_index + 1, img.size[0], img.size[1], max_input_size
                )

        final_size = list(img.size)
        resolution = final_size

        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        img_path = out_dir_path / f"page_{page_index}.png"
        img.save(img_path)

        elapsed_sec = time.time() - t0

        logger.info(
            "[RENDER][P%03d] PDF: %.1fx%.1fpt → DPI%d: %dx%d → Final: %dx%d | rot=%s | %.3fs",
            page_index + 1,
            pdf_width_pt,
            pdf_height_pt,
            dpi,
            rendered_size[0],
            rendered_size[1],
            final_size[0],
            final_size[1],
            page_rotation,
            elapsed_sec,
        )

        meta = {
            "pdf_width_pt": pdf_width_pt,
            "pdf_height_pt": pdf_height_pt,
            "dpi": dpi,
            "rendered_size": rendered_size,
            "final_size": final_size,
            "page_rotation": page_rotation,
            "detected_rotation": detected_rotation,
            "was_rotated": was_rotated,
            "was_resized": was_resized,
            "time_sec": elapsed_sec,
        }

        return img_path, resolution, meta

    finally:
        doc.close()


def render_pdf_pages(
    pdf_path: Path | str,
    dpi: int,
    out_dir: Path | str,
    logger: logging.Logger,
    page_indices: Optional[List[int]] = None,
    max_input_size: Optional[int] = None,
) -> List[Path]:
    """Render selected PDF pages to images, returning list of image paths."""
    total_pages = get_pdf_page_count(pdf_path)

    if page_indices is not None:
        pages_to_render = sorted(set(idx for idx in page_indices if 0 <= idx < total_pages))
        logger.info("Rendering %d/%d selected pages", len(pages_to_render), total_pages)
    else:
        pages_to_render = list(range(total_pages))
        logger.info("Rendering all %d pages", total_pages)

    image_paths: List[Path] = []
    for page_index in pages_to_render:
        img_path, _, _ = render_one_pdf_page(
            pdf_path=pdf_path,
            page_index=page_index,
            dpi=dpi,
            out_dir=out_dir,
            logger=logger,
            max_input_size=max_input_size,
            apply_rotation_detection=True,
        )
        image_paths.append(img_path)

    return image_paths


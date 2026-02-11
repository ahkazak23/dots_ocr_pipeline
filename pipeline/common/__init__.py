#!/usr/bin/env python3
"""Common pipeline utilities for OCR processing."""

# Page selection
from .page_select import parse_page_selection

# File system utilities
from .fs import (
    build_selection_label,
    build_page_output_dir,
    build_page_json_path,
    build_page_debug_dir,
    ensure_dir,
    atomic_write_json,
    read_json_if_exists,
    validate_page_payload_minimal,
)

# Spec payload builder
from .spec import build_spec_page_payload

# Image preprocessing
from .image_preprocess import (
    detect_page_rotation,
    resize_image_if_needed,
    normalize_image_rgb,
    preprocess_image,
)

# Postprocessing
from .postprocess import (
    normalize_text,
    classify_header_footer_heuristic,
)

# PDF rendering
from .pdf_render import (
    get_pdf_page_count,
    render_one_pdf_page,
    render_pdf_pages,
)

__all__ = [
    # Page selection
    "parse_page_selection",

    # File system
    "build_selection_label",
    "build_page_output_dir",
    "build_page_json_path",
    "build_page_debug_dir",
    "ensure_dir",
    "atomic_write_json",
    "read_json_if_exists",
    "validate_page_payload_minimal",

    # Spec
    "build_spec_page_payload",

    # Image preprocessing
    "detect_page_rotation",
    "resize_image_if_needed",
    "normalize_image_rgb",
    "preprocess_image",

    # Postprocessing
    "normalize_text",
    "classify_header_footer_heuristic",

    # PDF rendering
    "get_pdf_page_count",
    "render_one_pdf_page",
    "render_pdf_pages",
]


#!/usr/bin/env python3
"""
DeepSeek-OCR pipeline: extract structured text/tables from PDFs and images.

Processes documents page-by-page using DeepSeek-OCR with grounded bounding boxes.
Uses 2-attempt fallback: base inference, then optional crop_mode if extraction fails.
Outputs per-page spec-compliant JSON (page_<N>.json), optional legacy document.json,
and run summary with per-page timing in evaluation_output/.

Supports HuggingFace and Ollama backends.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import tempfile
import time
import warnings
import logging
import enum
import base64
import urllib.request
import urllib.error
import threading
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# Try to import pytesseract for rotation detection
try:
    import pytesseract

    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    pytesseract = None

# Suppress all transformer warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)

try:
    from colorama import init as colorama_init, Fore, Style

    colorama_init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    # Fallback if colorama not installed
    class _DummyColor:
        def __getattr__(self, name):
            return ""


    Fore = _DummyColor()
    Style = _DummyColor()
    COLORS_AVAILABLE = False

# Constants

REF_RE = re.compile(
    r"<\|ref\|>(?P<label>.*?)<\|/ref\|>\s*<\|det\|>(?P<det>\[.*?\])<\|/det\|>\s*",
    re.DOTALL,
)

LABEL_TO_TYPE = {
    "text": "text_block",
    "title": "title",
    "table": "table",
    "table_caption": "table",
    "image": "image",
    "diagram": "diagram",
    "header": "page_header",
    "footer": "page_footer",
    "page-header": "page_header",
    "page-footer": "page_footer",
}

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

MAX_IMAGE_DIMENSION = 9999  # HuggingFace: 2000, Ollama: 1280 works better

DEFAULT_PROMPT_EXTRACT_ALL = "<image>\n<|grounding|>Free OCR."

PROMPT = DEFAULT_PROMPT_EXTRACT_ALL

# Mode-specific inference configurations
INFERENCE_MODES = {
    "extract_all": {
        "prompt": "<image>\n<|grounding|>Free OCR.",
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,  # Enable crop mode for edge detection
    },
    "document": {
        "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,  # Enable crop mode for edge detection
    },
    "chart": {
        "prompt": "<image>\n<|grounding|>Parse all charts and tables.",
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,  # Enable crop mode for edge detection
    },
    "dense_text": {
        "prompt": "<image>\n<|grounding|>Free OCR.",
        "base_size": 768,
        "image_size": 512,
        "crop_mode": True,  # Enable crop mode for edge detection
    },
    "high_detail": {
        "prompt": "<image>\n<|grounding|>Extract complete document.",
        "base_size": 1024,
        "image_size": 768,
        "crop_mode": True,  # Enable crop mode for edge detection
    },
    "multilingual": {
        "prompt": "<image>\n<|grounding|>Extract text.",
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,  # Enable crop mode for edge detection
    },
    "experimental_tiny": {
        "prompt": "<image>\n<|grounding|>Convert to markdown.",
        "base_size": 512,
        "image_size": 384,
        "crop_mode": True,  # Enable crop mode for edge detection
    },
    "experimental_large": {
        "prompt": "<image>\n<|grounding|>Convert to markdown.",
        "base_size": 1280,
        "image_size": 896,
        "crop_mode": True,  # Enable crop mode for edge detection
    },
    "experimental_simple_prompt": {
        "prompt": "<image>\n<|grounding|>Extract all text.",
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,  # Enable crop mode for edge detection
    },
    "experimental_detailed_prompt": {
        "prompt": "<image>\n<|grounding|>Extract all content.",
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,  # Enable crop mode for edge detection
    },
    "experimental_base_768_img_640": {
        "prompt": "<image>\n<|grounding|>Convert to markdown.",
        "base_size": 768,
        "image_size": 640,
        "crop_mode": True,  # Enable crop mode for edge detection
    },
    "experimental_base_1024_img_512": {
        "prompt": "<image>\n<|grounding|>Convert to markdown.",
        "base_size": 1024,
        "image_size": 512,
        "crop_mode": True,  # Enable crop mode for edge detection
    },
    "experimental_base_896_img_768": {
        "prompt": "<image>\n<|grounding|>Convert to markdown.",
        "base_size": 896,
        "image_size": 768,
        "crop_mode": True,  # Enable crop mode for edge detection
    },
    "layout_first": {
        "prompt": "<image>\n<|grounding|>Detect layout.",
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,  # Enable crop mode for edge detection
    },
}

INFERENCE_CONFIG = {
    "base_size": 1024,
    "image_size": 640,
    "crop_mode": True,  # Enable crop mode for edge detection
    "test_compress": False,
}


# Logging

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    LEVEL_COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        if COLORS_AVAILABLE:
            level_color = self.LEVEL_COLORS.get(record.levelno, "")
            record.levelname = f"{level_color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


def _setup_logger(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("deepseek_ocr")
    if logger.handlers:
        return logger
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)

    if COLORS_AVAILABLE:
        fmt = ColoredFormatter("%(asctime)s | %(levelname)s | %(message)s")
    else:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.propagate = False
    return logger


# Backend

class Backend(str, enum.Enum):
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


# Config

@dataclass
class Config:
    """Configuration for DeepSeek-OCR pipeline."""
    model: str = "deepseek-ai/DeepSeek-OCR"
    backend: Backend = Backend.HUGGINGFACE
    dpi: int = 200
    out_dir: str = "./out_json/deepseek_ocr"
    eval_out_dir: str = "./evaluation_output"
    log_level: str = "INFO"
    debug_keep_renders: bool = True
    save_bbox_overlay: bool = True
    disable_fallbacks: bool = False
    test_all_modes: bool = False  # Run all inference modes on each page
    disable_downscale: bool = False  # Disable MAX_IMAGE_DIMENSION downscaling for edge detection testing
    store_raw_metadata: bool = False  # Store raw detection metadata (raw_label, raw_det, etc.) for debugging


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek model loading."""
    device: str = "auto"
    revision: Optional[str] = None


def is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def is_image(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def resolve_device(device_arg: str, logger: logging.Logger) -> Tuple[str, str]:
    # Device only matters for HuggingFace backend
    arg = device_arg.lower()
    if arg == "cuda":
        if torch.cuda.is_available():
            return "cuda", "cuda"
        logger.warning("CUDA requested but not available; falling back to CPU")
        logger.warning("Inference will be significantly slower (~2-5 min/page vs ~10-30 sec/page)")
        logger.info(
            "To enable GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        return "cpu", "cpu"
    if arg == "mps":
        if torch.backends.mps.is_available():
            return "mps", "mps"
        logger.warning("MPS requested but not available; falling back to CPU")
        return "cpu", "cpu"
    if arg == "cpu":
        return "cpu", "cpu"
    # Auto mode: cuda > mps > cpu
    if torch.cuda.is_available():
        return "cuda", "cuda"
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon GPU)")
        return "mps", "mps"
    logger.warning("No GPU available; using CPU")
    logger.warning("Inference will be significantly slower (~2-5 min/page vs ~10-30 sec/page)")
    logger.info(
        "To enable GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    return "cpu", "cpu"


def build_model(model_name: str, device: str, revision: Optional[str], logger: logging.Logger, backend: Backend) -> \
        Tuple[Any, Any]:
    if backend == Backend.OLLAMA:
        logger.info("Using Ollama backend; model is managed by Ollama daemon")
        return None, None

    logger.info("Loading model %s on %s", model_name, device)
    logger.info("Note: first run will download ~6GB model files from HuggingFace")

    tokenizer_kwargs = {"trust_remote_code": True}
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "use_safetensors": True,
        "torch_dtype": torch.bfloat16 if device in ("cuda", "mps") else torch.float32,
    }
    if revision:
        tokenizer_kwargs["revision"] = revision
        model_kwargs["revision"] = revision

    if device == "cuda":
        model_kwargs["device_map"] = {"": 0}
    elif device == "mps":
        model_kwargs["device_map"] = "mps"
    else:
        model_kwargs["device_map"] = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    model = AutoModel.from_pretrained(model_name, **model_kwargs).eval()
    logger.info("Model loaded")
    return model, tokenizer


def detect_page_rotation(img: Image.Image, logger: logging.Logger) -> int:
    """
    Detect page rotation using Tesseract OSD (Orientation and Script Detection).

    Args:
        img: PIL Image to analyze
        logger: Logger instance

    Returns:
        Rotation angle in degrees (0, 90, 180, or 270)
    """
    if not PYTESSERACT_AVAILABLE:
        logger.debug("pytesseract not available; skipping rotation detection")
        return 0

    try:
        # Use Tesseract's OSD to detect orientation
        osd_data = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        rotation = osd_data.get("rotate", 0)
        confidence = osd_data.get("orientation_conf", 0.0)

        logger.debug("[ROTATION] Detected rotation=%d° (confidence=%.1f%%)", rotation, confidence)

        # Only trust high-confidence detections
        if confidence < 1.0:
            logger.debug("[ROTATION] Low confidence (%.1f%%), skipping rotation", confidence)
            return 0

        return int(rotation)
    except Exception as exc:
        logger.debug("[ROTATION] Detection failed: %s: %s", type(exc).__name__, exc)
        return 0


def render_pdf_to_images(
        pdf_path: Path,
        dpi: int,
        tmp_dir: Path,
        logger: logging.Logger,
        page_indices: Optional[List[int]] = None,
        debug_save_dir: Optional[Path] = None,
        disable_downscale: bool = False,
) -> List[Path]:
    """Render selected PDF pages to images.

    Args:
        pdf_path: Input PDF path.
        dpi: Render DPI.
        tmp_dir: Output directory for rendered page images.
        page_indices: Optional list of 0-based page indices to render.
        debug_save_dir: If provided, save intermediate transformation images here.
        disable_downscale: If True, skip MAX_IMAGE_DIMENSION downscaling.

    Returns:
        List of rendered image paths.
    """
    doc = fitz.open(pdf_path)
    image_paths: List[Path] = []

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    total_pages = len(doc)
    if page_indices is not None:
        pages_to_render = sorted(set(idx for idx in page_indices if 0 <= idx < total_pages))
        logger.info("Rendering %d/%d selected pages", len(pages_to_render), total_pages)
    else:
        pages_to_render = list(range(total_pages))
        logger.info("Rendering all %d pages", total_pages)

    for page_index in pages_to_render:
        page = doc[page_index]
        t0 = time.time()
        try:
            rotation = int(page.rotation)
        except Exception:
            rotation = None

        # Get original PDF page dimensions
        page_rect = page.rect
        pdf_width_pt = page_rect.width
        pdf_height_pt = page_rect.height

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        mode = "RGB" if pixmap.n < 4 else "RGBA"
        img = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
        if img.mode != "RGB":
            img = img.convert("RGB")

        original_size = img.size
        was_downscaled = False
        was_rotated = False
        detected_rotation = 0

        # Detect and correct page rotation using Tesseract OSD
        detected_rotation = detect_page_rotation(img, logger)
        if detected_rotation > 0:
            # Rotate image to correct orientation
            # Tesseract returns counterclockwise rotation needed, PIL rotates counterclockwise by default
            img = img.rotate(-detected_rotation, expand=True, resample=Image.Resampling.BICUBIC)
            was_rotated = True
            logger.info(
                "[RENDER][P%03d] Rotation correction: detected=%d° → rotated %d° (new size: %dx%d)",
                page_index + 1,
                detected_rotation,
                -detected_rotation,
                img.size[0],
                img.size[1]
            )

        # Save Step 1: Original render (after DPI scaling, before MAX_IMAGE_DIMENSION)
        if debug_save_dir:
            step1_path = debug_save_dir / f"step1_dpi_render_{img.size[0]}x{img.size[1]}.png"
            img.save(step1_path)
            logger.debug(
                "[RENDER][P%03d] Step 1 saved: DPI render → %s",
                page_index + 1,
                step1_path.name
            )

        if max(img.size) > MAX_IMAGE_DIMENSION and not disable_downscale:
            original_size = img.size
            scale = MAX_IMAGE_DIMENSION / max(img.size)
            new_size = (int(img.width * scale), int(img.height * scale))
            resample = getattr(Image, "LANCZOS", None) or getattr(Image.Resampling, "LANCZOS", 1)
            img = img.resize(new_size, resample)
            was_downscaled = True
            logger.info(
                "[RENDER][P%03d] Step 2: Downscaled %s → %s (%.1f%% scale, max_dim=%d)",
                page_index + 1,
                original_size,
                new_size,
                (new_size[0] / original_size[0]) * 100,
                MAX_IMAGE_DIMENSION,
            )

            # Save Step 2: After MAX_IMAGE_DIMENSION downscaling
            if debug_save_dir:
                step2_path = debug_save_dir / f"step2_max_dim_{img.size[0]}x{img.size[1]}.png"
                img.save(step2_path)
                logger.debug(
                    "[RENDER][P%03d] Step 2 saved: MAX_IMAGE_DIMENSION downscale → %s",
                    page_index + 1,
                    step2_path.name
                )

        img_path = tmp_dir / f"page_{page_index}.png"
        img.save(img_path)
        image_paths.append(img_path)

        elapsed_sec = time.time() - t0
        logger.info(
            "[RENDER][P%03d] PDF: %.1fx%.1fpt → DPI%d: %dx%d → Final: %dx%d | rot=%s | %.3fs",
            page_index + 1,
            pdf_width_pt,
            pdf_height_pt,
            dpi,
            pixmap.width,
            pixmap.height,
            img.size[0],
            img.size[1],
            rotation,
            elapsed_sec,
        )

    doc.close()
    return image_paths


def _parse_det(det_text: str):
    det_text = det_text.strip()
    try:
        return ast.literal_eval(det_text)
    except Exception:
        try:
            return json.loads(det_text)
        except Exception:
            return None


def _det_to_bbox_normalized(det_coords, img_w: int, img_h: int) -> List[float]:
    """Convert detection coordinates to normalized bbox [x1, y1, x2, y2] in 0-1 range."""
    if not det_coords or not isinstance(det_coords, list):
        return [0.0, 0.0, 0.0, 0.0]

    boxes = [b for b in det_coords if isinstance(b, (list, tuple)) and len(b) == 4]
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]

    xs1, ys1, xs2, ys2 = zip(*boxes)
    x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)

    denom = 999.0 if max(x2, y2) <= 999 else 1000.0

    px1 = x1 / denom * img_w
    py1 = y1 / denom * img_h
    px2 = x2 / denom * img_w
    py2 = y2 / denom * img_h

    px1 = max(0, min(px1, img_w))
    py1 = max(0, min(py1, img_h))
    px2 = max(0, min(px2, img_w))
    py2 = max(0, min(py2, img_h))

    norm_x1 = px1 / img_w if img_w > 0 else 0.0
    norm_y1 = py1 / img_h if img_h > 0 else 0.0
    norm_x2 = px2 / img_w if img_w > 0 else 0.0
    norm_y2 = py2 / img_h if img_h > 0 else 0.0

    return [
        max(0.0, min(1.0, norm_x1)),
        max(0.0, min(1.0, norm_y1)),
        max(0.0, min(1.0, norm_x2)),
        max(0.0, min(1.0, norm_y2)),
    ]


def classify_header_footer_heuristic(
        blocks: List[Dict[str, Any]],
        logger: Optional[logging.Logger] = None
) -> List[Dict[str, Any]]:
    """
    Classify blocks as page-header or page-footer based on heuristics.

    Heuristics:
    - Top 10% of page → likely header
    - Bottom 10% of page → likely footer
    - Contains page numbers (e.g., "Page 1", "1 of 3", "- 1 -")
    - Small height relative to page (< 5%)
    - Contains typical header/footer keywords

    Returns updated blocks with corrected types.
    """
    HEADER_THRESHOLD = 0.10  # Top 10% of page
    FOOTER_THRESHOLD = 0.90  # Bottom 10% of page
    MAX_HEIGHT_RATIO = 0.05  # Max 5% of page height

    FOOTER_PATTERNS = [
        r'\bpage\s+\d+\b',  # "page 1", "Page 2"
        r'\b\d+\s+of\s+\d+\b',  # "1 of 3"
        r'^-?\s*\d+\s*-?$',  # "- 1 -", "1"
        # Removed year pattern r'\b\d{4}\b' - causes false positives on document dates
        r'confidential|proprietary|copyright|©',  # Legal text
    ]

    HEADER_PATTERNS = [
        r'^(chapter|section)\s+\d+',  # "Chapter 1"
        r'confidential|draft|internal',
    ]

    if not blocks:
        return blocks

    updated_blocks = []
    header_count = 0
    footer_count = 0

    for block in blocks:
        bbox = block.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            updated_blocks.append(block)
            continue

        x1, y1, x2, y2 = bbox
        block_type = block.get("type", "paragraph")
        text = block.get("extraction_response", "")

        # Calculate width and height for heuristics
        w = x2 - x1
        h = y2 - y1

        # Skip blocks with zero bbox (corrupted/malformed output)
        if bbox == [0.0, 0.0, 0.0, 0.0]:
            if logger:
                logger.debug(
                    "[HEURISTIC] Skipping zero-bbox block (corrupted): type=%s, text=%s",
                    block_type, text[:50] if text else ""
                )
            updated_blocks.append(block)
            continue

        # Skip if already classified
        if block_type in ["page-header", "page-footer"]:
            updated_blocks.append(block)
            continue

        # Skip tables and images from header/footer classification
        if block_type in ["table", "image", "diagram"]:
            updated_blocks.append(block)
            continue

        # Check position-based heuristics
        is_top_region = y1 < HEADER_THRESHOLD
        is_bottom_region = y2 > FOOTER_THRESHOLD
        is_small_height = h < MAX_HEIGHT_RATIO

        # Check content patterns
        text_lower = text.lower() if text else ""
        has_footer_pattern = any(re.search(pat, text_lower, re.IGNORECASE) for pat in FOOTER_PATTERNS)
        has_header_pattern = any(re.search(pat, text_lower, re.IGNORECASE) for pat in HEADER_PATTERNS)

        # Classification logic
        new_type = block_type
        reason = None

        if is_bottom_region and is_small_height:
            new_type = "page-footer"
            reason = f"bottom region (y2={y2:.3f}) + small height (h={h:.3f})"
            footer_count += 1
        elif is_bottom_region and has_footer_pattern:
            new_type = "page-footer"
            reason = f"bottom region + footer pattern in text"
            footer_count += 1
        elif is_top_region and is_small_height:
            new_type = "page-header"
            reason = f"top region (y1={y1:.3f}) + small height (h={h:.3f})"
            header_count += 1
        elif is_top_region and has_header_pattern:
            new_type = "page-header"
            reason = f"top region + header pattern in text"
            header_count += 1
        elif has_footer_pattern and not is_top_region:
            # Page number anywhere except top
            new_type = "page-footer"
            reason = "footer pattern (page number/copyright)"
            footer_count += 1

        if new_type != block_type:
            if logger:
                logger.debug(
                    "[HEURISTIC] Block reclassified: %s → %s | bbox=[%.3f, %.3f, %.3f, %.3f] | reason: %s | text: %s",
                    block_type, new_type, x1, y1, x2, y2, reason, text[:50]
                )
            block = {**block, "type": new_type}

        updated_blocks.append(block)

    if logger and (header_count > 0 or footer_count > 0):
        logger.info(
            "[HEURISTIC] Classified %d headers, %d footers from %d blocks",
            header_count, footer_count, len(blocks)
        )

    return updated_blocks


def normalize_text(text: str) -> str:
    """Normalize text: strip outer whitespace, normalize newlines, collapse excessive spaces per line."""
    if not text:
        return ""
    # Normalize Windows newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip outer whitespace
    text = text.strip()
    # Collapse multiple spaces on each line (preserve line breaks)
    lines = text.split("\n")
    normalized_lines = [" ".join(line.split()) for line in lines]
    return "\n".join(normalized_lines)


def try_parse_markdown_table(md: str) -> Optional[Dict[str, Any]]:
    """
    Parse markdown table into structured data.
    Returns {"columns": [...], "rows": [[...], ...]} or None if parsing fails.
    Fail-safe: if separator/header is missing or malformed, return None.
    If parse succeeds but rows have fewer cells, pad with "".
    """
    if not md or "|" not in md:
        return None

    lines = [line.strip() for line in md.split("\n") if line.strip()]
    if len(lines) < 2:
        return None

    # Find separator line (contains dashes and pipes)
    separator_idx = -1
    for i, line in enumerate(lines):
        if "|" in line and "-" in line:
            # Check if it looks like a separator (mostly dashes/pipes/colons/spaces)
            cleaned = line.replace("|", "").replace("-", "").replace(":", "").strip()
            if not cleaned or all(c in "- |:" for c in line):
                separator_idx = i
                break

    if separator_idx < 1:  # Need at least header before separator
        return None

    # Parse header
    header_line = lines[separator_idx - 1]
    columns = [cell.strip() for cell in header_line.split("|")]
    # Remove empty leading/trailing cells from pipes at start/end
    if columns and not columns[0]:
        columns = columns[1:]
    if columns and not columns[-1]:
        columns = columns[:-1]

    if not columns:
        return None

    col_count = len(columns)

    # Parse data rows (after separator)
    rows = []
    for i in range(separator_idx + 1, len(lines)):
        row_line = lines[i]
        if "|" not in row_line:
            continue
        cells = [cell.strip() for cell in row_line.split("|")]
        # Remove empty leading/trailing cells
        if cells and not cells[0]:
            cells = cells[1:]
        if cells and not cells[-1]:
            cells = cells[:-1]

        # Pad row to match column count
        while len(cells) < col_count:
            cells.append("")
        # Truncate if too many cells
        cells = cells[:col_count]

        rows.append(cells)

    return {"columns": columns, "rows": rows}


def try_parse_html_table(html: str) -> Optional[Dict[str, Any]]:
    """
    Parse HTML table into structured data.
    Returns {"headers": [...], "rows": [[...], ...]} or None if parsing fails.
    """
    if not html or "<table>" not in html.lower():
        return None

    # Simple regex-based HTML table parsing
    # Extract all <tr> tags
    tr_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.IGNORECASE | re.DOTALL)
    tr_matches = tr_pattern.findall(html)

    if not tr_matches:
        return None

    headers = []
    rows = []

    for i, tr_content in enumerate(tr_matches):
        # Check if this is a header row (contains <th> tags)
        th_pattern = re.compile(r'<th[^>]*>(.*?)</th>', re.IGNORECASE | re.DOTALL)
        td_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.IGNORECASE | re.DOTALL)

        th_matches = th_pattern.findall(tr_content)
        td_matches = td_pattern.findall(tr_content)

        if th_matches and i == 0:
            # First row with <th> tags is the header
            headers = [_clean_html(cell) for cell in th_matches]
        elif td_matches:
            # Data row
            row = [_clean_html(cell) for cell in td_matches]
            rows.append(row)
        elif th_matches and not headers:
            # Fallback: treat <th> as headers even if not first row
            headers = [_clean_html(cell) for cell in th_matches]

    # If we found headers and rows, return structured data
    if headers or rows:
        return {"headers": headers, "rows": rows}

    return None


def _clean_html(text: str) -> str:
    """Remove HTML tags and clean up text."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Decode HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&amp;', '&')
    text = text.replace('&quot;', '"')
    # Clean up whitespace
    text = ' '.join(text.split())
    return text.strip()


def build_spec_page_payload(
        document_id: str,
        page_id: str,
        resolution: List[int],
        blocks: List[Dict[str, Any]],
        warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a per-page spec payload matching OmniDocBench format."""
    # Convert page_id to integer
    try:
        page_id_int = int(page_id)
    except (ValueError, TypeError):
        page_id_int = 0

    payload = {
        "meta": {
            "document_id": document_id,
            "page_id": page_id_int,
            "page_resolution": resolution,
            "warnings": warnings if warnings else [],
        },
        "ocr": {"blocks": blocks},
    }

    if warnings:
        payload["warnings"] = warnings

    return payload


def parse_grounded_stdout(
        captured: str,
        image_path: Path,
        logger: logging.Logger,
        page_id: str = "",
        store_raw: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[int]]:
    """Parse grounded model output.

    Args:
        captured: Raw model output text
        image_path: Path to the image file
        logger: Logger instance
        page_id: Page identifier for logging context
        store_raw: If True, store raw detection metadata for debugging

    Returns:
        (legacy_page_ocr, parse_diagnostics, resolution)
    """
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    resolution = [img_w, img_h]

    logger.debug("[%s] Parsing grounded output: image=%s (%dx%d), captured_len=%d",
                 page_id or "UNKNOWN", image_path.name, img_w, img_h, len(captured))

    matches = list(REF_RE.finditer(captured))
    blocks: List[Dict[str, Any]] = []
    bbox_parse_failures = 0
    malformed_tags = 0

    # Check for malformed grounding tags (incomplete <|det|> or corrupted output)
    if "<|ref|>" in captured:
        # Count incomplete tags
        ref_count = captured.count("<|ref|>")
        det_count = captured.count("<|det|>")
        det_close_count = captured.count("<|/det|>")

        if ref_count != det_count or det_count != det_close_count:
            logger.warning(
                "[MALFORMED] Unbalanced grounding tags on %s: ref=%d, det=%d, /det=%d",
                image_path.name, ref_count, det_count, det_close_count
            )
            malformed_tags = abs(ref_count - det_count) + abs(det_count - det_close_count)

        # Check for "eee" pattern (signature redaction artifact)
        if "eee" in captured.lower():
            eee_count = captured.lower().count("eee")
            if eee_count > 3:
                logger.warning(
                    "[MALFORMED] Detected 'eee' pattern (%d occurrences) - likely redacted signatures/watermarks",
                    eee_count
                )

    for i, match in enumerate(matches):
        label = match.group("label").strip()
        det_text = match.group("det").strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(captured)
        content = captured[start:end].strip()

        # Check for corrupted content
        is_corrupted = (
                not content or
                content.startswith("]]<|/det|>") or
                "eee eee" in content.lower() or
                len(content) < 2
        )

        if is_corrupted:
            logger.warning(
                "[MALFORMED] Block %d has corrupted content: label=%s, content_preview=%s",
                len(blocks), label, content[:100]
            )

        try:
            det_coords = _parse_det(det_text)
            bbox = _det_to_bbox_normalized(det_coords, img_w, img_h)

            # Additional check: if bbox is all zeros but det_text exists, log detailed info
            if bbox == [0.0, 0.0, 0.0, 0.0] and det_text:
                logger.warning(
                    "[MALFORMED] Zero bbox from det_text=%s | label=%s | content_preview=%s",
                    det_text[:100], label, content[:50]
                )
        except Exception as exc:
            bbox = [0.0, 0.0, 0.0, 0.0]
            bbox_parse_failures += 1
            logger.warning(
                "bbox parse failed for block %d on %s: %s: %s | det_text=%s",
                len(blocks),
                image_path.name,
                type(exc).__name__,
                exc,
                det_text[:100],
            )

        block_type = LABEL_TO_TYPE.get(label, "text_block")

        if block_type == "table":
            # Try parsing HTML table first, then fall back to Markdown
            table_data = try_parse_html_table(content)
            if table_data is None:
                table_data = try_parse_markdown_table(content)
            parsed_payload = {"data": table_data, "text": normalize_text(content)}
        elif block_type == "image":
            parsed_payload = {"data": None, "text": ""}
        else:
            parsed_payload = {"data": None, "text": normalize_text(content)}

        # Build block matching OmniDocBench format
        # Keep id/order for backward compatibility with bbox overlay tools
        legacy_block = {
            "type": block_type,
            "bbox": bbox,
            "extraction_origin": "deepseek-ocr",
            "extraction_response": content,
            "extraction_response_parsed": parsed_payload,
            # Legacy fields for bbox overlay compatibility
            "id": f"blk_{len(blocks) + 1}",
            "order": len(blocks),
        }

        # Add raw metadata for debugging if requested
        if store_raw:
            try:
                det_coords_parsed = _parse_det(det_text) if det_text else None
            except:
                det_coords_parsed = None

            legacy_block["raw_label"] = label
            legacy_block["raw_det"] = det_text
            legacy_block["raw_det_parsed"] = det_coords_parsed
            legacy_block["raw_bbox_parse_failed"] = bbox == [0.0, 0.0, 0.0, 0.0] and bool(det_text)
            legacy_block["raw_match_index"] = i
            legacy_block["raw_is_corrupted"] = is_corrupted

        blocks.append(legacy_block)

    parse_diag = {
        "img_w": img_w,
        "img_h": img_h,
        "match_count": len(matches),
        "block_count": len(blocks),
        "bbox_parse_failures": bbox_parse_failures,
        "malformed_tags": malformed_tags,
    }

    # Return legacy format for bbox overlay, diagnostics, and resolution
    return {"ocr": {"blocks": blocks}}, parse_diag, resolution


# Helper functions for inference

def _has_grounding_tags(text: str) -> bool:
    """Check if text contains DeepSeek grounding tags."""
    return bool(text and "<|ref|>" in text and "<|det|>" in text)


def _run_inference(
        model: Any,
        tokenizer: Any,
        image_path: Path,
        infer_config: Dict[str, Any],
        timeout_seconds: int = 600,  # Increased from 60s to prevent truncated outputs
        debug_save_path: Optional[Path] = None,  # Path to save raw output for debugging
) -> Tuple[str, float]:
    """Run model inference and capture output with timeout.

    Args:
        model: The model instance
        tokenizer: The tokenizer instance
        image_path: Path to input image
        infer_config: Inference configuration dictionary
        timeout_seconds: Maximum time to wait for inference
        debug_save_path: If provided, save raw model output to this file

    Returns:
        (captured_text, inference_time_sec)
    """
    logger = logging.getLogger("deepseek_ocr")

    t0 = time.time()
    # Create fresh StringIO buffers for this inference
    out_buf = StringIO()
    err_buf = StringIO()

    prompt = infer_config.get("prompt", PROMPT)
    config_for_infer = {k: v for k, v in infer_config.items() if k != "prompt"}

    # Log input image dimensions and model config
    try:
        img = Image.open(image_path)
        img_w, img_h = img.size
        logger.info(
            "[INFERENCE] Input: %s (%dx%d px) | Model config: base_size=%s, image_size=%s",
            image_path.name,
            img_w,
            img_h,
            config_for_infer.get("base_size", "default"),
            config_for_infer.get("image_size", "default"),
        )
    except Exception as e:
        logger.warning("[INFERENCE] Could not read input image dimensions: %s", e)

    logger.debug("Inference config: %s", {k: v for k, v in config_for_infer.items() if k != "prompt"})
    logger.debug("Timeout: %ds", timeout_seconds)

    res = None
    exception_msg = None
    timeout_occurred = False

    # Save original stdout/stderr to restore later
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    def _inference_worker():
        """Worker function to run inference in a thread."""
        nonlocal res, exception_msg
        try:
            # Directly replace sys.stdout/stderr for more reliable capture
            sys.stdout = out_buf
            sys.stderr = err_buf

            with torch.inference_mode():
                res = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=str(image_path),
                    output_path=str(image_path.parent),
                    save_results=False,
                    **config_for_infer,
                )
        except Exception as exc:
            exception_msg = f"{type(exc).__name__}: {exc}"
            # Restore stdout/stderr before logging
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            logger.error("Inference exception in worker: %s", exception_msg)
        finally:
            # Always restore stdout/stderr when worker thread exits
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    # Start inference in a separate thread (non-daemon to ensure cleanup)
    logger.debug("Starting model.infer() in background thread with %ds timeout...", timeout_seconds)
    worker_thread = threading.Thread(target=_inference_worker, daemon=False)
    worker_thread.start()

    # Wait for completion or timeout
    worker_thread.join(timeout=timeout_seconds)

    if worker_thread.is_alive():
        # Thread is still running - timeout occurred
        timeout_occurred = True
        exception_msg = f"TimeoutException: Inference exceeded {timeout_seconds}s limit"
        logger.error("Inference TIMEOUT after %ds (thread still running)", timeout_seconds)
        # Give thread a brief moment to write any buffered output
        time.sleep(0.5)
        # Note: We can't actually kill the thread, but we can stop waiting for it
    else:
        logger.debug("model.infer() completed within timeout")

    # Ensure stdout/stderr are restored in main thread (in case of early return or timeout)
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    infer_time = time.time() - t0
    logger.debug("Inference wall time: %.2fs", infer_time)

    # Flush buffers to ensure all content is captured
    try:
        out_buf.flush()
        err_buf.flush()
    except Exception:
        pass

    captured = ""
    if isinstance(res, str) and res.strip():
        captured = res.strip()
        logger.debug("Captured from return value: %d chars", len(captured))
    elif out_buf.getvalue().strip():
        captured = out_buf.getvalue().strip()
        logger.debug("Captured from stdout: %d chars", len(captured))
    elif err_buf.getvalue().strip():
        captured = err_buf.getvalue().strip()
        logger.debug("Captured from stderr: %d chars", len(captured))
    else:
        logger.debug("No output captured")

    if exception_msg:
        if not captured:
            captured = f"[INFERENCE_EXCEPTION] {exception_msg}"

    return captured, infer_time


def _run_ollama_inference(
        image_path: Path,
        logger: logging.Logger,
        prompt: str,
        model_name: str = "deepseek-ocr",
) -> Tuple[str, float]:
    """Run inference via Ollama HTTP API with an image."""
    t0 = time.time()
    try:
        img_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

        payload = {"model": model_name, "prompt": prompt, "images": [img_b64], "stream": False}
        data = json.dumps(payload).encode("utf-8")

        host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        req = urllib.request.Request(
            url=f"{host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            resp_data = resp.read().decode("utf-8")
            try:
                obj = json.loads(resp_data)
                captured = obj.get("response", "")
            except Exception:
                captured = resp_data

        infer_time = time.time() - t0
        return str(captured).strip(), infer_time

    except urllib.error.HTTPError as exc:
        infer_time = time.time() - t0
        logger.error("Ollama HTTP error %s: %s", exc.code, exc.reason)
        return "", infer_time
    except urllib.error.URLError as exc:
        infer_time = time.time() - t0
        host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        logger.error("Ollama not reachable at %s: %s", host, exc.reason)
        logger.error("Ensure 'ollama serve' is running and port 11434 is free.")
        return "", infer_time
    except Exception as exc:
        infer_time = time.time() - t0
        logger.error("Ollama inference failed: %s: %s", type(exc).__name__, exc)
        return "", infer_time


def run_page_ocr(
        model: Any,
        tokenizer: Any,
        image_path: Path,
        infer_config: Dict[str, Any],
        logger: logging.Logger,
        page_id: str = "",
        disable_fallbacks: bool = False,
        backend: Backend = Backend.HUGGINGFACE,
        test_all_modes: bool = False,
        page_dir: Optional[Path] = None,
        store_raw: bool = False,
) -> Tuple[Dict[str, Any], float, str, List[int]]:
    """Run OCR on a single rendered page image.

    Args:
        test_all_modes: If True, run all INFERENCE_MODES on this page and log timing.
        page_dir: Output directory for the page (used for saving mode outputs in test_all_modes)
        store_raw: If True, store raw detection metadata for debugging

    Returns:
        (page_ocr, inference_time_sec, attempt_used, resolution)
    """
    if test_all_modes and backend == Backend.HUGGINGFACE:
        return _run_all_modes_experiment(model, tokenizer, image_path, logger, page_id, page_dir)

    if backend == Backend.OLLAMA:
        logger.info("[%s] Inference (ollama)", page_id)
        prompt = infer_config.get("prompt", DEFAULT_PROMPT_EXTRACT_ALL)
        captured, infer_time = _run_ollama_inference(image_path, logger, prompt=prompt)
        if _has_grounding_tags(captured):
            page_ocr, _, resolution = parse_grounded_stdout(captured, image_path, logger, page_id=page_id,
                                                            store_raw=store_raw)
            return page_ocr, infer_time, "ollama", resolution
        logger.warning("[%s] No grounding tags detected", page_id)
        img = Image.open(image_path).convert("RGB")
        resolution = [img.size[0], img.size[1]]
        return {"ocr": {"blocks": []}}, infer_time, "none", resolution

    logger.info("[%s] Inference (base)", page_id)
    debug_path = page_dir / f"{page_id.replace('|', '_')}_raw_base.txt" if page_dir else None
    captured, time_base = _run_inference(model, tokenizer, image_path, infer_config, debug_save_path=debug_path)

    logger.debug("[%s] Base attempt captured %d chars", page_id, len(captured))
    if captured:
        preview = captured[:200] if len(captured) > 200 else captured
        logger.debug("[%s] Base output preview: %r", page_id, preview)
    else:
        logger.warning("[%s] Base attempt produced empty output", page_id)

    if _has_grounding_tags(captured):
        logger.info("[%s] Inference succeeded (base)", page_id)
        page_ocr, _, resolution = parse_grounded_stdout(captured, image_path, logger, page_id=page_id,
                                                        store_raw=store_raw)
        return page_ocr, time_base, "base", resolution

    if not disable_fallbacks:
        logger.info("[%s] Inference (crop_mode fallback)", page_id)
        crop_config = dict(infer_config)
        crop_config["crop_mode"] = True
        debug_path_crop = page_dir / f"{page_id.replace('|', '_')}_raw_crop.txt" if page_dir else None
        captured_crop, time_crop = _run_inference(model, tokenizer, image_path, crop_config,
                                                  debug_save_path=debug_path_crop)

        logger.debug("[%s] Crop attempt captured %d chars", page_id, len(captured_crop))
        if captured_crop:
            preview_crop = captured_crop[:200] if len(captured_crop) > 200 else captured_crop
            logger.debug("[%s] Crop output preview: %r", page_id, preview_crop)
        else:
            logger.warning("[%s] Crop attempt produced empty output", page_id)

        if _has_grounding_tags(captured_crop):
            logger.warning("[%s] Base attempt failed; crop_mode fallback succeeded", page_id)
            page_ocr, _, resolution = parse_grounded_stdout(captured_crop, image_path, logger, page_id=page_id,
                                                            store_raw=store_raw)
            return page_ocr, time_base + time_crop, "crop_mode", resolution

        total_time = time_base + time_crop
        logger.warning("[%s] No grounding tags detected after all attempts", page_id)
        img = Image.open(image_path).convert("RGB")
        resolution = [img.size[0], img.size[1]]
        return {"ocr": {"blocks": []}}, total_time, "none", resolution

    logger.warning("[%s] Base attempt failed; fallbacks disabled", page_id)
    img = Image.open(image_path).convert("RGB")
    resolution = [img.size[0], img.size[1]]
    return {"ocr": {"blocks": []}}, time_base, "none", resolution


def _run_all_modes_experiment(
        model: Any,
        tokenizer: Any,
        image_path: Path,
        logger: logging.Logger,
        page_id: str,
        page_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], float, str, List[int]]:
    """Run all inference modes on a page and log timing results.

    Returns the result from the first successful mode.
    """
    logger.info("[%s] === RUNNING ALL MODES EXPERIMENT ===", page_id)
    logger.info("[%s] Image path: %s", page_id, image_path)

    img = Image.open(image_path).convert("RGB")
    resolution = [img.size[0], img.size[1]]
    logger.info("[%s] Image resolution: %dx%d", page_id, resolution[0], resolution[1])

    # Create output directory for raw outputs
    # Use persistent directory under page_dir if provided
    if page_dir is not None:
        mode_out_dir = page_dir / "mode_outputs"
    else:
        mode_out_dir = image_path.parent / f"{page_id.replace('|', '_')}_mode_outputs"

    mode_out_dir.mkdir(parents=True, exist_ok=True)

    # Log CWD to make debugging easier
    logger.info("[%s] CWD=%s", page_id, os.getcwd())
    logger.info("[%s] Raw outputs will be saved to: %s", page_id, str(mode_out_dir.resolve()))

    results = []
    first_success = None
    total_time = 0.0

    for idx, (mode_name, mode_config) in enumerate(INFERENCE_MODES.items(), 1):
        logger.info("[%s] " + "=" * 60, page_id)
        logger.info("[%s] Testing mode %d/%d: %s", page_id, idx, len(INFERENCE_MODES), mode_name)
        logger.debug("[%s]   base_size: %d", page_id, mode_config.get("base_size", 1024))
        logger.debug("[%s]   image_size: %d", page_id, mode_config.get("image_size", 640))
        logger.debug("[%s]   crop_mode: %s", page_id, mode_config.get("crop_mode", False))
        prompt_text = mode_config.get("prompt", "")
        if len(prompt_text) > 100:
            logger.debug("[%s]   prompt: %s...", page_id, prompt_text[:100])
        else:
            logger.debug("[%s]   prompt: %s", page_id, prompt_text)

        test_config = dict(mode_config)

        try:
            logger.debug("[%s] Starting inference for mode: %s", page_id, mode_name)
            t_start = time.time()
            captured, infer_time = _run_inference(model, tokenizer, image_path, test_config, timeout_seconds=60)
            t_end = time.time()
            total_time += infer_time

            logger.debug("[%s] Inference completed in %.2fs (wall time: %.2fs)", page_id, infer_time, t_end - t_start)

            success = _has_grounding_tags(captured)
            char_count = len(captured)

            # Count grounding tags
            ref_count = captured.count("<|ref|>") if captured else 0
            det_count = captured.count("<|det|>") if captured else 0

            logger.debug("[%s] Output length: %d chars", page_id, char_count)
            logger.debug("[%s] Grounding tags found: <|ref|>=%d, <|det|>=%d", page_id, ref_count, det_count)

            # Save raw output
            raw_output_file = mode_out_dir / f"{mode_name}.txt"
            raw_output_file.write_text(captured if captured else "[EMPTY OUTPUT]", encoding="utf-8")
            logger.debug("[%s] Raw output saved to: %s", page_id, raw_output_file.name)

            # Parse and save bbox image for successful modes
            mode_blocks = []
            if success:
                try:
                    parsed_ocr, _, _ = parse_grounded_stdout(captured, image_path, logger, page_id=page_id)
                    mode_blocks = parsed_ocr.get("ocr", {}).get("blocks", [])

                    # Save bbox image for this mode
                    bbox_img_path = mode_out_dir / f"{mode_name}_bbox.png"
                    save_bbox_image(image_path, mode_blocks, bbox_img_path)
                    logger.debug("[%s] Bbox image saved for mode %s: %d blocks", page_id, mode_name, len(mode_blocks))
                except Exception as bbox_exc:
                    logger.warning("[%s] Failed to save bbox for mode %s: %s", page_id, mode_name, bbox_exc)

            # Save parsed JSON result
            mode_result = {
                "mode": mode_name,
                "success": success,
                "char_count": char_count,
                "ref_tag_count": ref_count,
                "det_tag_count": det_count,
                "block_count": len(mode_blocks),
            }
            json_output_file = mode_out_dir / f"{mode_name}.json"
            json_output_file.write_text(
                json.dumps(mode_result, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            result_entry = {
                "mode": mode_name,
                "time_sec": round(infer_time, 2),
                "success": success,
                "char_count": char_count,
                "ref_tag_count": ref_count,
                "det_tag_count": det_count,
                "block_count": len(mode_blocks),
                "base_size": mode_config.get("base_size", 1024),
                "image_size": mode_config.get("image_size", 640),
                "prompt": mode_config.get("prompt", ""),
                "raw_output_file": str(raw_output_file.name),
                "bbox_image": f"{mode_name}_bbox.png" if success else None,
            }
            results.append(result_entry)

            status = "✓ SUCCESS" if success else "✗ FAILED"
            if success:
                logger.info("[%s] %s | %s | time=%.1fs | chars=%d | tags=%d | blocks=%d",
                            page_id, mode_name.ljust(30), status, infer_time, char_count, ref_count, len(mode_blocks))
            else:
                logger.warning("[%s] %s | %s | time=%.1fs | chars=%d | tags=%d",
                               page_id, mode_name.ljust(30), status, infer_time, char_count, ref_count)

            if success and first_success is None:
                logger.info("[%s] First successful mode: %s", page_id, mode_name)
                page_ocr_dict = {"ocr": {"blocks": mode_blocks}}
                first_success = (page_ocr_dict, result_entry["mode"])

        except Exception as exc:
            logger.error("[%s] Mode %s exception: %s: %s", page_id, mode_name, type(exc).__name__, exc)
            import traceback
            logger.debug("[%s] Traceback:\n%s", page_id, traceback.format_exc())
            results.append({
                "mode": mode_name,
                "time_sec": 0.0,
                "success": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            })

    # Log summary
    logger.info("[%s] " + "=" * 60, page_id)
    logger.info("[%s] === EXPERIMENT SUMMARY ===", page_id)
    logger.info("[%s] Total modes tested: %d", page_id, len(results))
    logger.info("[%s] Total time: %.1fs (%.1f min)", page_id, total_time, total_time / 60)

    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]

    if successful:
        logger.info("[%s] Successful modes: %d / %d", page_id, len(successful), len(results))
        fastest = min(successful, key=lambda x: x["time_sec"])
        slowest = max(successful, key=lambda x: x["time_sec"])
        logger.info("[%s] Fastest: %s (%.1fs)", page_id, fastest["mode"], fastest["time_sec"])
        logger.info("[%s] Slowest: %s (%.1fs)", page_id, slowest["mode"], slowest["time_sec"])
        logger.info("[%s] Speed ratio: %.1fx", page_id, slowest["time_sec"] / fastest["time_sec"])
    else:
        logger.warning("[%s] No modes succeeded! All %d modes failed.", page_id, len(failed))

    if failed:
        logger.warning("[%s] Failed modes: %d", page_id, len(failed))
        for r in failed:
            if "error" in r:
                logger.debug("[%s]   - %s: %s", page_id, r["mode"], r.get("error", "unknown"))

    # Save results to JSON (in mode_out_dir)
    results_file = mode_out_dir / "mode_comparison.json"
    results_file.write_text(json.dumps({
        "page_id": page_id,
        "image_path": str(image_path),
        "resolution": resolution,
        "total_time_sec": round(total_time, 2),
        "successful_count": len(successful),
        "failed_count": len(failed),
        "results": results,
    }, indent=2), encoding="utf-8")
    logger.info("[%s] Mode comparison JSON saved to: %s", page_id, str(results_file.resolve()))
    logger.info("[%s] " + "=" * 60, page_id)

    if first_success:
        page_ocr, mode_used = first_success
        return page_ocr, total_time, f"experiment_{mode_used}", resolution
    else:
        return {"ocr": {"blocks": []}}, total_time, "experiment_none", resolution


def save_bbox_image(page_img_path: Path, blocks: List[Dict[str, Any]], dest_path: Path) -> None:
    """Draw bounding boxes on page image and save."""
    from PIL import ImageDraw

    img = Image.open(page_img_path).convert("RGB")
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)

    for blk in blocks:
        bbox = blk.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue

        norm_x1, norm_y1, norm_x2, norm_y2 = bbox

        # Calculate corners with rounding (not truncation) for accuracy
        x1 = int(round(norm_x1 * img_w))
        y1 = int(round(norm_y1 * img_h))
        x2 = int(round(norm_x2 * img_w))
        y2 = int(round(norm_y2 * img_h))

        # Clamp to image bounds to prevent out-of-bounds drawing
        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(0, min(img_w, x2))
        y2 = max(0, min(img_h, y2))

        # Ensure at least 1 pixel wide/tall (prevents zero-size boxes)
        if x2 <= x1:
            x2 = min(img_w, x1 + 1)
        if y2 <= y1:
            y2 = min(img_h, y1 + 1)

        # Draw rectangle (skip if still invalid)
        try:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        except Exception:
            pass

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest_path)


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).convert("RGB").save(dst)


def process_document(
        model: Any,
        tokenizer: Any,
        input_path: Path,
        cfg: Config,
        output_root: Path,
        selection_label: str,
        logger: logging.Logger,
        pages: Optional[List[int]] = None,
        page_range: Optional[str] = None,
        backend: Backend = Backend.HUGGINGFACE,
        inference_config: Optional[Dict[str, Any]] = None,
        legacy_document_json: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Process one input document and write per-page outputs.

    Returns:
        (legacy_doc_json, metrics_summary)
    """
    warnings_list: List[str] = []
    input_type = "pdf" if is_pdf(input_path) else "image"

    # OmniDocBench format: use subdirectory only for page selections
    if selection_label and selection_label != "all_pages":
        selection_dir = output_root / selection_label
    else:
        selection_dir = output_root
    selection_dir.mkdir(parents=True, exist_ok=True)

    # Create separate debug directory for artifacts
    debug_dir = None
    if cfg.debug_keep_renders or cfg.save_bbox_overlay:
        debug_dir = selection_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

    infer_config = inference_config if inference_config is not None else INFERENCE_CONFIG

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmp_dir = Path(tmpdir_str)

        if input_type == "pdf":
            pdf_doc = fitz.open(input_path)
            total_pages = len(pdf_doc)
            pdf_doc.close()

            selected_indices = parse_page_selection(pages, page_range, total_pages)

            # Create debug directory for transformation images if debug is enabled
            debug_transforms_dir = None
            if cfg.debug_keep_renders:
                debug_transforms_dir = debug_dir / "transforms" if debug_dir else selection_dir / "debug" / "transforms"
                debug_transforms_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Debug mode: Saving intermediate transformation images to %s", debug_transforms_dir)

            page_image_paths = render_pdf_to_images(
                input_path, cfg.dpi, tmp_dir, logger, page_indices=selected_indices,
                debug_save_dir=debug_transforms_dir, disable_downscale=cfg.disable_downscale
            )
        else:
            page_image_paths = [input_path]
            selected_indices = parse_page_selection(pages, page_range, len(page_image_paths))

        pages_out: List[Dict[str, Any]] = []
        total_time = 0.0

        if input_type == "pdf":
            page_index_to_position = {idx: pos for pos, idx in enumerate(selected_indices)}
        else:
            page_index_to_position = {0: 0}

        for page_index in selected_indices:
            position = page_index_to_position[page_index]
            page_path = page_image_paths[position]

            # OmniDocBench format: no per-page subdirectories
            page_id = f"{input_path.stem}|P{page_index:04d}"

            if cfg.debug_keep_renders and debug_dir:
                try:
                    _copy_image(page_path, debug_dir / f"page_{page_index:04d}_render.png")
                except Exception as exc:
                    logger.warning("[%s] failed to persist render.png: %s: %s", page_id, type(exc).__name__, exc)

            try:
                page_ocr, infer_time, attempt_used, resolution = run_page_ocr(
                    model=model,
                    tokenizer=tokenizer,
                    image_path=page_path,
                    infer_config=infer_config,
                    logger=logger,
                    page_id=page_id,
                    disable_fallbacks=cfg.disable_fallbacks,
                    backend=cfg.backend,
                    test_all_modes=cfg.test_all_modes,
                    page_dir=debug_dir,  # Pass debug_dir for any page-specific debug files
                    store_raw=cfg.store_raw_metadata,
                )

                if cfg.save_bbox_overlay and debug_dir:
                    try:
                        save_bbox_image(page_path, page_ocr.get("ocr", {}).get("blocks", []),
                                        debug_dir / f"page_{page_index:04d}_bbox.png")
                    except Exception as exc:
                        logger.warning("[%s] failed to save bbox overlay: %s: %s", page_id, type(exc).__name__, exc)

                document_id = input_path.stem
                page_id_str = str(page_index)

                # Apply heuristic header/footer classification
                raw_blocks = page_ocr.get("ocr", {}).get("blocks", [])
                classified_blocks = classify_header_footer_heuristic(raw_blocks, logger)

                spec_blocks: List[Dict[str, Any]] = []

                for blk in classified_blocks:
                    block_type = blk.get("type", "paragraph")
                    bbox = blk.get("bbox", [0, 0, 0, 0])

                    spec_block = {
                        "type": block_type,
                        "bbox": bbox,
                        "extraction_origin": blk.get("extraction_origin", "deepseek-ocr"),
                        "extraction_response": blk["extraction_response"],
                        "extraction_response_parsed": blk["extraction_response_parsed"],
                    }

                    # Preserve raw metadata fields if present
                    if cfg.store_raw_metadata:
                        for key in [
                            "raw_label", "raw_det", "raw_det_parsed", "raw_bbox_parse_failed", "raw_match_index",
                            "raw_is_corrupted"]:
                            if key in blk:
                                spec_block[key] = blk[key]

                    spec_blocks.append(spec_block)

                page_warnings: List[str] = []
                if attempt_used == "none":
                    page_warnings.append("OCR extraction failed - no grounding tags detected")
                elif attempt_used == "crop_mode":
                    page_warnings.append("Base attempt failed, used crop_mode fallback")

                if len(spec_blocks) == 0 and attempt_used != "none":
                    page_warnings.append("No blocks extracted from page")

                spec_payload = build_spec_page_payload(
                    document_id=document_id,
                    page_id=page_id_str,
                    resolution=resolution,
                    blocks=spec_blocks,
                    warnings=page_warnings if page_warnings else None,
                )

                # OmniDocBench format: page_XXXX.json with 0-based 4-digit padding
                page_json_path = selection_dir / f"page_{page_index:04d}.json"
                page_json_path.write_text(json.dumps(spec_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.debug("[%s] wrote %s", page_id, page_json_path.name)

                page_payload = {
                    "meta": {
                        "model": cfg.model,
                        "input_file": str(input_path),
                        "input_type": input_type,
                        "page_index": page_index,
                        "page_label": page_index + 1,
                        "page_id": page_id,
                    },
                    "page_ocr": page_ocr,
                    "metrics": {"time_sec": round(infer_time, 4)},
                    "diagnostics": {"attempt_used": attempt_used},
                }

                pages_out.append(page_payload)
                total_time += infer_time

            except Exception as exc:
                warnings_list.append(f"Page {page_index} failed: {type(exc).__name__}: {exc}")
                logger.error("[%s] page failed: %s: %s", page_id, type(exc).__name__, exc)

                try:
                    img = Image.open(page_path).convert("RGB")
                    resolution = [img.size[0], img.size[1]]
                except Exception:
                    resolution = [0, 0]

                document_id = input_path.stem
                page_id_str = str(page_index)
                error_warnings = [f"Page processing failed: {type(exc).__name__}: {exc}"]
                spec_payload = build_spec_page_payload(
                    document_id=document_id,
                    page_id=page_id_str,
                    resolution=resolution,
                    blocks=[],
                    warnings=error_warnings,
                )
                # OmniDocBench format: page_XXXX.json with 0-based 4-digit padding
                page_json_path = selection_dir / f"page_{page_index:04d}.json"
                page_json_path.write_text(json.dumps(spec_payload, ensure_ascii=False, indent=2), encoding="utf-8")

                page_payload = {
                    "meta": {
                        "model": cfg.model,
                        "input_file": str(input_path),
                        "input_type": input_type,
                        "page_index": page_index,
                        "page_label": page_index + 1,
                        "page_id": page_id,
                    },
                    "page_ocr": {"ocr": {"blocks": []}},
                    "metrics": {"time_sec": None},
                    "diagnostics": {"error": f"{type(exc).__name__}: {exc}"},
                }
                pages_out.append(page_payload)

    page_count = len(pages_out)

    page_timings: List[Dict[str, Any]] = []
    for page_payload in pages_out:
        meta = page_payload.get("meta", {})
        metrics = page_payload.get("metrics", {})
        diagnostics = page_payload.get("diagnostics", {})
        ocr = page_payload.get("page_ocr", {}).get("ocr", {})

        page_timings.append(
            {
                "page_index": meta.get("page_index"),
                "page_label": meta.get("page_label"),
                "time_sec": metrics.get("time_sec"),
                "attempt_used": diagnostics.get("attempt_used"),
                "blocks_extracted": len(ocr.get("blocks", [])),
                "error": diagnostics.get("error"),
            }
        )

    summary = {
        "total_pages": page_count,
        "total_time_sec": round(total_time, 4),
        "avg_time_per_page_sec": round(total_time / page_count, 4) if page_count else 0.0,
        "page_timings": page_timings,
    }

    doc_json = {
        "meta": {
            "model": cfg.model,
            "input_file": str(input_path),
            "input_type": input_type,
            "page_count": page_count,
            "dpi": cfg.dpi if input_type == "pdf" else None,
            "warnings": warnings_list,
            "output_dir": str(selection_dir),
        },
        "pages": pages_out,
        "metrics_summary": summary,
    }

    if legacy_document_json:
        out_path = selection_dir / "document.json"
        out_path.write_text(json.dumps(doc_json, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Legacy document.json written to %s", out_path)

    return doc_json, summary


def gather_inputs(input_arg: Path) -> List[Path]:
    if input_arg.is_file():
        return [input_arg]
    if input_arg.is_dir():
        candidates = []
        for path in sorted(input_arg.rglob("*")):
            if path.is_file() and (is_pdf(path) or is_image(path)):
                candidates.append(path)
        return candidates
    raise FileNotFoundError(f"Input not found: {input_arg}")


def write_run_summary(
        eval_dir: Path,
        run_items: List[Dict[str, Any]],
        total_time: float,
        total_pages: int,
        device: str,
        model: str,
) -> Path:
    eval_dir.mkdir(parents=True, exist_ok=True)
    ts_filename = time.strftime("%Y%m%d_%H%M%S")
    ts_readable = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path = eval_dir / f"deepseek_ocr_run_{ts_filename}.json"
    payload = {
        "model": model,
        "device": device,
        "run_started": ts_readable,
        "total_documents": len(run_items),
        "total_pages": total_pages,
        "total_time_sec": round(total_time, 4),
        "documents": run_items,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-OCR pipeline")
    parser.add_argument("--input", required=True, help="Path to a PDF, image, or folder of files")
    parser.add_argument("--output-dir", default="./out_json/deepseek_ocr", help="Output directory for JSON files")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF rendering")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Device selection")
    parser.add_argument("--revision", default=None, help="Optional model revision")
    parser.add_argument("--pages", type=int, nargs="+", help="Page indices to process (1-based, e.g., 1 5 10)")
    parser.add_argument("--page-range", type=str, help="Inclusive page range (1-based, e.g., 1-10)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Console log level")
    parser.add_argument("--keep-renders", action="store_true",
                        help="Persist rendered page images under page folders (recommended)")
    parser.add_argument(
        "--no-bbox-overlay",
        action="store_true",
        default=False,
        help="Disable bounding box overlay images (enabled by default)",
    )
    parser.add_argument(
        "--enable-fallbacks",
        action="store_true",
        help="Enable crop_mode fallback when base attempt fails (may cause CUDA errors)",
    )
    parser.add_argument(
        "--disable-fallbacks",
        action="store_true",
        help="[DEPRECATED] Use --enable-fallbacks instead. Fallbacks are now disabled by default.",
    )
    parser.add_argument("--backend", default="huggingface", choices=[b.value for b in Backend],
                        help="Backend: huggingface or ollama")
    parser.add_argument(
        "--mode",
        default="extract_all",
        choices=list(INFERENCE_MODES.keys()),
        help="Inference mode with optimized settings (extract_all is default)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom prompt to override mode default (e.g., '<image>\\n<|grounding|>Extract all text.')",
    )
    parser.add_argument("--test-compress", action="store_true",
                        help="Enable test_compress mode for compression diagnostics")
    parser.add_argument("--base-size", type=int, default=None, help="Override base_size (e.g., 512, 768, 1024, 1280)")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Override image_size (e.g., 384, 512, 640, 768, 896)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Maximum tokens to generate (default: model default)")
    parser.add_argument(
        "--legacy-document-json",
        action="store_true",
        default=False,
        help="Write legacy document.json in addition to per-page spec files (default: OFF)",
    )
    parser.add_argument(
        "--test-all-modes",
        action="store_true",
        default=False,
        help="Run all inference modes on each page with 3-minute timeout per mode (for debugging slowdowns)",
    )
    parser.add_argument(
        "--disable-downscale",
        action="store_true",
        default=False,
        help="Disable MAX_IMAGE_DIMENSION downscaling (keeps full resolution for edge detection testing)",
    )
    return parser.parse_args(argv)


def parse_page_selection(pages: Optional[List[int]], page_range: Optional[str], total_pages: int) -> List[int]:
    def normalize(idx: int) -> int:
        return idx - 1

    if pages:
        return [normalize(p) for p in pages if 1 <= p <= total_pages]
    if page_range:
        try:
            start_s, end_s = page_range.split("-")
            start, end = int(start_s), int(end_s)
            start = max(start, 1)
            end = min(end, total_pages)
            if start <= end:
                return list(range(normalize(start), normalize(end) + 1))
        except ValueError:
            pass
    return list(range(total_pages))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logger = _setup_logger(args.log_level)

    backend = Backend(args.backend)
    if backend == Backend.HUGGINGFACE:
        _, resolved_device = resolve_device(args.device, logger)
    else:
        resolved_device = "cpu"

    cfg = Config(
        backend=backend,
        dpi=args.dpi,
        out_dir=args.output_dir,
        log_level=args.log_level,
        debug_keep_renders=bool(args.keep_renders),
        save_bbox_overlay=not args.no_bbox_overlay,
        disable_fallbacks=not args.enable_fallbacks if not args.disable_fallbacks else args.disable_fallbacks,
        test_all_modes=args.test_all_modes,
        disable_downscale=args.disable_downscale,
        store_raw_metadata=args.raw,
    )
    ds_cfg = DeepSeekConfig(device=resolved_device, revision=args.revision)

    input_path = Path(args.input).expanduser().resolve()
    try:
        inputs = gather_inputs(input_path)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 2
    if not inputs:
        logger.error("No valid PDF or image files found")
        return 2

    model, tokenizer = build_model(cfg.model, resolved_device, ds_cfg.revision, logger, backend=backend)

    inference_config = INFERENCE_MODES.get(args.mode, INFERENCE_MODES["extract_all"]).copy()

    if args.prompt is not None:
        inference_config["prompt"] = args.prompt
        logger.info("Prompt: custom (--prompt)")
    else:
        logger.info("Prompt: mode default (%s)", args.mode)

    if args.test_compress:
        inference_config["test_compress"] = True
        logger.info("test_compress enabled")

    if args.base_size:
        inference_config["base_size"] = args.base_size
        logger.info("base_size overridden to %d", args.base_size)
    if args.image_size:
        inference_config["image_size"] = args.image_size
        logger.info("image_size overridden to %d", args.image_size)
    if args.max_new_tokens:
        inference_config["max_new_tokens"] = args.max_new_tokens
        logger.info("max_new_tokens set to %d", args.max_new_tokens)

    if cfg.disable_fallbacks:
        logger.info("Crop mode fallback: disabled")
    else:
        logger.info("Crop mode fallback: enabled")

    logger.info(
        "Mode: %s (base_size=%s, image_size=%s)",
        args.mode,
        inference_config.get("base_size"),
        inference_config.get("image_size"),
    )

    if args.legacy_document_json:
        logger.info("Legacy document.json output: enabled")
    else:
        logger.info("Legacy document.json output: disabled")

    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = Path(cfg.eval_out_dir).expanduser().resolve()

    run_items: List[Dict[str, Any]] = []
    grand_total_time = 0.0
    total_pages = 0

    run_start = time.time()
    for input_file in inputs:
        selection_label = "all_pages"
        if args.pages:
            selection_label = "pages_" + "-".join(map(str, args.pages))
        elif args.page_range:
            selection_label = f"pages_{args.page_range}"

        doc_root = out_dir / input_file.stem
        doc_root.mkdir(parents=True, exist_ok=True)

        logger.info("Processing document: %s (selection=%s)", input_file.name, selection_label)

        doc_json, summary = process_document(
            model,
            tokenizer,
            input_file,
            cfg,
            output_root=doc_root,
            selection_label=selection_label,
            logger=logger,
            pages=args.pages,
            page_range=args.page_range,
            inference_config=inference_config,
            legacy_document_json=args.legacy_document_json,
        )

        selection_dir = doc_root / selection_label
        out_path = selection_dir / "document.json" if args.legacy_document_json else selection_dir

        run_items.append(
            {
                "input_file": str(input_file),
                "output_file": str(out_path),
                "pages": summary["total_pages"],
                "total_time_sec": summary["total_time_sec"],
                "avg_time_per_page_sec": summary["avg_time_per_page_sec"],
                "page_timings": summary["page_timings"],
            }
        )
        grand_total_time += summary["total_time_sec"]
        total_pages += summary["total_pages"]

    run_total_time = time.time() - run_start
    summary_path = write_run_summary(
        eval_dir=eval_dir,
        run_items=run_items,
        total_time=grand_total_time,
        total_pages=total_pages,
        device=resolved_device,
        model=cfg.model,
    )

    logger.info("Processed %d documents; output_dir=%s", len(inputs), out_dir)
    logger.info("Run summary saved to %s", summary_path)
    logger.info("Elapsed wall time: %.2fs (model load included)", run_total_time)
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    sys.exit(main())


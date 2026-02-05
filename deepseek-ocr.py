#!/usr/bin/env python3
"""
DeepSeek-OCR pipeline: extract structured text/tables from PDFs and images.

Processes documents page-by-page using DeepSeek-OCR with grounded bounding boxes.
Runs single-pass inference (no crop_mode fallback).
Outputs per-page spec-compliant JSON (page_<N>.json), optional legacy document.json,
and run summary with per-page timing in evaluation_output/.
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
    r"<\|ref\|>(?P<label>.*?)<\|/ref\|>.*?<\|det\|>(?P<det>\[.*?\])<\|/det\|>\s*",
    re.DOTALL,
)

LABEL_TO_TYPE = {
    "text": "text_block",
    "title": "title",
    "table": "table",
    "table_caption": "table_caption",  # Keep separate for post-processing
    "image": "image",
    "diagram": "diagram",
    "header": "page_header",
    "footer": "page_footer",
    "page-header": "page_header",
    "page-footer": "page_footer",
}

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

DEFAULT_PROMPT_EXTRACT_ALL = "<image>\n<|grounding|>Convert the document to markdown."

PROMPT = DEFAULT_PROMPT_EXTRACT_ALL


INFERENCE_CONFIG = {
    "prompt": "<image>\n<|grounding|>Free OCR",
    "base_size": 1024,
    "image_size": 640,
    "crop_mode" : True,
    "test_compress": True,  # Match Colab default behavior
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


# Config

@dataclass
class Config:
    """Configuration for DeepSeek-OCR pipeline."""
    model: str = "deepseek-ai/DeepSeek-OCR"
    dpi: int = 200
    out_dir: str = "./out_json/deepseek_ocr"
    eval_out_dir: str = "./evaluation_output"
    log_level: str = "INFO"
    store_raw_metadata: bool = False  # Store raw detection metadata (raw_label, raw_det, etc.) for debugging


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek model loading."""
    device: str = "auto"


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


def build_model(model_name: str, device: str, logger: logging.Logger) -> \
        Tuple[Any, Any]:

    logger.info("Loading model %s on %s", model_name, device)
    logger.info("Note: first run will download ~6GB model files from HuggingFace")

    # Detect GPU capabilities for flash attention and dtype selection
    gpu_major, gpu_minor = None, None
    if device == "cuda" and torch.cuda.is_available():
        gpu_major, gpu_minor = torch.cuda.get_device_capability(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)} ({gpu_mem_gb:.1f} GB) - Compute Capability: SM{gpu_major}{gpu_minor}")

    # Determine dtype based on GPU capability (Ampere+ = bf16, older = fp16)
    if device == "cuda" and torch.cuda.is_available():
        # Only use bfloat16 on Ampere+ (SM 8.0+) where it's natively supported
        if gpu_major >= 8 and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            logger.info("Using bfloat16 (Ampere+ GPU with native BF16 support)")
        else:
            torch_dtype = torch.float16
            logger.info(f"Using float16 (SM{gpu_major}{gpu_minor} GPU, BF16 not natively supported)")
    elif device == "mps":
        torch_dtype = torch.float16
        logger.info("Using float16 for MPS device")
    else:
        torch_dtype = torch.float32
        logger.info("Using float32 for CPU")

    # Check for flash attention availability on Ampere+ GPUs
    use_flash_attn = False
    flash_attn_working = False
    if device == "cuda" and gpu_major is not None and gpu_major >= 8:
        try:
            # Try importing flash_attn to check if it's working
            import flash_attn
            from flash_attn.flash_attn_interface import flash_attn_func
            flash_attn_working = True
            use_flash_attn = True
            logger.info("FlashAttention-2 detected and working")
        except (ImportError, OSError, AttributeError) as e:
            # Flash attention not available or broken
            if "ImportError" in str(type(e).__name__) and "flash_attn" not in str(e).lower():
                logger.warning(f"Flash attention installed but has import errors: {e}")
                logger.warning("Will use eager attention. To fix: pip uninstall flash-attn && pip install flash-attn --no-build-isolation")
            elif "flash_attn" not in str(e).lower():
                logger.warning(f"Flash attention check failed: {e}")
            else:
                logger.info("Flash attention not installed; using eager attention")
                logger.info("For faster inference (optional): pip install flash-attn --no-build-isolation")
            # Disable flash attention to prevent errors during model loading
            use_flash_attn = False
            flash_attn_working = False

    # Prepare model loading kwargs (following official DeepSeek-OCR example)
    tokenizer_kwargs = {"trust_remote_code": True}
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "use_safetensors": True,
        # NOTE: Do NOT set torch_dtype or device_map here, as per official example
        # We'll handle dtype/device after loading with .eval().cuda().to(dtype)
    }

    # Set attention implementation (use _attn_implementation with underscore, as per official example)
    if use_flash_attn and flash_attn_working:
        model_kwargs["_attn_implementation"] = "flash_attention_2"
        logger.info("Will use FlashAttention-2 for model")
    else:
        # Force eager attention to prevent flash_attn import errors
        model_kwargs["_attn_implementation"] = "eager"
        logger.info("Will use eager attention (standard PyTorch)")

    # Load model (following official example: no device_map, no torch_dtype in from_pretrained)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    logger.info("Tokenizer loaded, loading model (this may take a minute)...")
    model = AutoModel.from_pretrained(model_name, **model_kwargs)

    # Apply eval + device + dtype explicitly (official DeepSeek-OCR pattern)
    model = model.eval()
    if device == "cuda":
        model = model.cuda().to(torch_dtype)
        logger.info(f"Model loaded: eval().cuda().to({torch_dtype})")
    elif device == "mps":
        model = model.to("mps").to(torch_dtype)
        logger.info(f"Model loaded: eval().to('mps').to({torch_dtype})")
    else:
        model = model.to(torch_dtype)
        logger.info(f"Model loaded: eval().to({torch_dtype})")

    logger.info("Model ready for inference")
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

        # Log full OSD data for debugging
        logger.debug("[ROTATION] Tesseract OSD data: %s", osd_data)

        rotation = osd_data.get("rotate", 0)
        confidence = osd_data.get("orientation_conf", 0.0)

        logger.debug("[ROTATION] Detected rotation=%d° (confidence=%.1f%%)", rotation, confidence)

        # Only trust high-confidence detections (15+ out of ~100)
        if confidence < 50.0:
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
) -> List[Path]:
    """Render selected PDF pages to images.

    Args:
        pdf_path: Input PDF path.
        dpi: Render DPI.
        tmp_dir: Output directory for rendered page images.
        page_indices: Optional list of 0-based page indices to render.

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

        # Downscaling permanently disabled - images kept at full DPI resolution for better quality

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
        if block_type in ["page_header", "page_footer"]:
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
            new_type = "page_footer"
            reason = f"bottom region (y2={y2:.3f}) + small height (h={h:.3f})"
            footer_count += 1
        elif is_bottom_region and has_footer_pattern:
            new_type = "page_footer"
            reason = f"bottom region + footer pattern in text"
            footer_count += 1
        elif is_top_region and is_small_height:
            new_type = "page_header"
            reason = f"top region (y1={y1:.3f}) + small height (h={h:.3f})"
            header_count += 1
        elif is_top_region and has_header_pattern:
            new_type = "page_header"
            reason = f"top region + header pattern in text"
            header_count += 1
        elif has_footer_pattern and not is_top_region:
            # Page number anywhere except top
            new_type = "page_footer"
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


def _split_caption_and_table(content: str) -> Tuple[str, str, str]:
    """
    Split table_caption content into: (caption_text, table_markup, table_type).

    Returns:
        (caption_text, table_markup, table_type)
        table_type is "html", "markdown", or "" if no table found
    """
    if not content:
        return "", "", ""

    # Try HTML table first
    if "<table" in content.lower():
        parts = content.split("<table", 1)
        caption_text = parts[0].strip()
        table_html = "<table" + parts[1] if len(parts) > 1 else ""
        return caption_text, table_html, "html"

    # Try Markdown table (look for | separators)
    lines = content.split("\n")
    table_start_idx = -1

    for i, line in enumerate(lines):
        # Markdown table has pipes and typically a separator line with dashes
        if "|" in line:
            # Check if next line looks like separator (contains dashes)
            if i + 1 < len(lines) and "-" in lines[i + 1] and "|" in lines[i + 1]:
                table_start_idx = i
                break

    if table_start_idx >= 0:
        # Found markdown table
        caption_lines = lines[:table_start_idx]
        table_lines = lines[table_start_idx:]

        # Find where table ends (first line without |, or end of content)
        table_end_idx = len(table_lines)
        for i, line in enumerate(table_lines):
            if line.strip() and "|" not in line:
                table_end_idx = i
                break

        caption_text = "\n".join(caption_lines).strip()
        table_markdown = "\n".join(table_lines[:table_end_idx]).strip()
        return caption_text, table_markdown, "markdown"

    # No table found
    return content.strip(), "", ""


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
    if not html or "<table" not in html.lower():
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
    img = Image.open(image_path)
    # Keep PDF-rendered images behavior stable (rendered PNGs normalized to RGB).
    # For user-provided images, preserve original mode (no forced RGB conversion).
    if image_path.suffix.lower() == ".png" and image_path.name.startswith("page_") and img.mode != "RGB":
        img = img.convert("RGB")

    img_w, img_h = img.size
    resolution = [img_w, img_h]

    logger.debug(
        "[%s] Parsing grounded output: image=%s (%dx%d), captured_len=%d",
        page_id or "UNKNOWN",
        image_path.name,
        img_w,
        img_h,
        len(captured),
    )

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

        # Clean up malformed content that starts with garbage followed by ref tags
        # Example: "简陋<|/ref|>" from corrupted output like "<|ref|>table<|/ref|>简陋<|/ref|>"
        original_content = content
        content_had_leading_garbage = False

        # Remove all leading garbage + ref tags (both opening and closing)
        while content:
            # Check for pattern: text followed by <|/ref|> at the start
            # This happens when there's an extra closing tag with garbage before it
            ref_close_match = re.match(r'^([^<]*)<\|/ref\|>', content)
            if ref_close_match:
                garbage_text = ref_close_match.group(1)
                content = content[ref_close_match.end():].strip()
                content_had_leading_garbage = True
                continue

            # Also check for opening ref tags at the start
            if content.startswith("<|ref|>"):
                content = content[len("<|ref|>"):].strip()
                content_had_leading_garbage = True
                continue

            # Also check for closing ref tags at the start (without preceding text)
            if content.startswith("<|/ref|>"):
                content = content[len("<|/ref|>"):].strip()
                content_had_leading_garbage = True
                continue

            # No more leading garbage found
            break

        if content_had_leading_garbage:
            logger.warning(
                "[MALFORMED] Block %d: Content started with garbage+ref tags. "
                "Removed prefix '%s...' → remaining content_len=%d",
                len(blocks), original_content[:50], len(content)
            )
            malformed_tags += 1
        elif ("<|/ref|>" in content or "<|ref|>" in content) and content:
            # Ref tags embedded in middle of content - also indicates corruption
            logger.warning(
                "[MALFORMED] Block %d: Content contains embedded ref tags: '%s...'",
                len(blocks), content[:100]
            )
            malformed_tags += 1

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
            # Initialize with empty caption text - will be populated during post-processing
            parsed_payload = {"data": table_data, "text": ""}
        elif block_type == "table_caption":
            # table_caption content often contains: caption text + table (HTML or Markdown)
            # Split them using the helper function
            caption_text, table_markup, table_type = _split_caption_and_table(content)

            # Parse table data based on type
            table_data = None
            if table_type == "html":
                table_data = try_parse_html_table(table_markup)
            elif table_type == "markdown":
                table_data = try_parse_markdown_table(table_markup)

            # Store both caption and table data
            parsed_payload = {
                "data": table_data,
                "text": normalize_text(caption_text),
                "_has_table_html": bool(table_markup),  # Flag for post-processing
                "_table_html": table_markup  # Store the actual table markup for extraction_response
            }
        elif block_type == "image":
            parsed_payload = {"data": None, "text": ""}
        else:
            parsed_payload = {"data": None, "text": normalize_text(content)}

        # Build block matching OmniDocBench format
        legacy_block = {
            "type": block_type,
            "bbox": bbox,
            "extraction_origin": "deepseek-ocr",
            "extraction_response": content,
            "extraction_response_parsed": parsed_payload,
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

    # Post-process: separate table and caption payloads, keep both blocks
    # Common pattern: <table> (empty) followed by <table_caption> (has caption text + table HTML/MD)
    processed_blocks = []
    i = 0
    while i < len(blocks):
        block = blocks[i]

        # Case 1: table (empty) followed by table_caption (with caption + table data)
        # This is the REAL DeepSeek output pattern - KEEP BOTH BLOCKS
        if block["type"] == "table" and i + 1 < len(blocks) and blocks[i + 1]["type"] == "table_caption":
            table_block = block
            caption_block = blocks[i + 1]

            table_parsed = table_block.get("extraction_response_parsed", {})
            table_has_data = table_parsed.get("data") is not None

            caption_parsed = caption_block.get("extraction_response_parsed", {})
            caption_text = caption_parsed.get("text", "")
            caption_has_table_data = caption_parsed.get("data") is not None
            table_html_from_caption = caption_parsed.get("_table_html", "")

            # Move table data from caption to table block if table was empty
            if caption_has_table_data and not table_has_data:
                table_block["extraction_response_parsed"]["data"] = caption_parsed.get("data")
                # Set extraction_response to the table HTML/MD for downstream tools
                if table_html_from_caption:
                    table_block["extraction_response"] = table_html_from_caption
                logger.debug(
                    "[%s] Moved table data from caption to table block (table was empty)",
                    page_id or "UNKNOWN"
                )

            # Clean caption block: keep only caption text, remove table data
            caption_block["extraction_response_parsed"] = {"data": None, "text": caption_text}
            caption_block["extraction_response"] = caption_text

            # Remove internal flags
            if "_has_table_html" in caption_parsed:
                del caption_parsed["_has_table_html"]
            if "_table_html" in caption_parsed:
                del caption_parsed["_table_html"]

            # Keep BOTH blocks with their respective bboxes
            processed_blocks.append(table_block)
            processed_blocks.append(caption_block)
            i += 2
            logger.debug(
                "[%s] Separated table + caption: caption_text='%s', table_has_data=%s",
                page_id or "UNKNOWN", caption_text[:50] if caption_text else "",
                caption_has_table_data or table_has_data
            )

        # Case 2: table_caption followed by table (reverse order - less common)
        elif block["type"] == "table_caption" and i + 1 < len(blocks) and blocks[i + 1]["type"] == "table":
            caption_block = block
            table_block = blocks[i + 1]

            caption_parsed = caption_block.get("extraction_response_parsed", {})
            caption_text = caption_parsed.get("text", "")
            caption_has_table_data = caption_parsed.get("data") is not None
            table_html_from_caption = caption_parsed.get("_table_html", "")

            table_parsed = table_block.get("extraction_response_parsed", {})
            table_has_data = table_parsed.get("data") is not None

            # Move table data from caption to table if table was empty
            if caption_has_table_data and not table_has_data:
                table_block["extraction_response_parsed"]["data"] = caption_parsed.get("data")
                if table_html_from_caption:
                    table_block["extraction_response"] = table_html_from_caption
                logger.debug(
                    "[%s] Moved table data from caption to table (reverse order)",
                    page_id or "UNKNOWN"
                )

            # Clean caption block
            caption_block["extraction_response_parsed"] = {"data": None, "text": caption_text}
            caption_block["extraction_response"] = caption_text

            # Remove internal flags
            if "_has_table_html" in caption_parsed:
                del caption_parsed["_has_table_html"]
            if "_table_html" in caption_parsed:
                del caption_parsed["_table_html"]

            # Keep BOTH blocks
            processed_blocks.append(caption_block)
            processed_blocks.append(table_block)
            i += 2
            logger.debug(
                "[%s] Separated caption + table (reverse): caption='%s'",
                page_id or "UNKNOWN", caption_text[:50] if caption_text else ""
            )

        # Case 3: Standalone table_caption with table data -> convert to table
        elif block["type"] == "table_caption":
            caption_parsed = block.get("extraction_response_parsed", {})
            caption_has_table_data = caption_parsed.get("data") is not None

            if caption_has_table_data:
                # Convert to table since it has table data
                block["type"] = "table"
                table_html = caption_parsed.get("_table_html", "")
                if table_html:
                    block["extraction_response"] = table_html
                # Keep caption text in parsed.text field
                if "_has_table_html" in block["extraction_response_parsed"]:
                    del block["extraction_response_parsed"]["_has_table_html"]
                if "_table_html" in block["extraction_response_parsed"]:
                    del block["extraction_response_parsed"]["_table_html"]
                processed_blocks.append(block)
                logger.debug(
                    "[%s] Converted standalone table_caption to table: caption=%s",
                    page_id or "UNKNOWN",
                    caption_parsed.get("text", "")[:50]
                )
            else:
                # Standalone caption without table - keep as caption or skip
                logger.warning(
                    "[%s] Found standalone table_caption without table data: %s",
                    page_id or "UNKNOWN",
                    caption_parsed.get("text", "")[:50]
                )
                # Keep it as-is (some documents have standalone captions)
                processed_blocks.append(block)
            i += 1

        # For all other blocks, keep as-is
        else:
            processed_blocks.append(block)
            i += 1

    blocks = processed_blocks

    parse_diag = {
        "img_w": img_w,
        "img_h": img_h,
        "match_count": len(matches),
        "block_count": len(blocks),
        "bbox_parse_failures": bbox_parse_failures,
        "malformed_tags": malformed_tags,
    }

    # Return legacy format, diagnostics, and resolution
    return {"ocr": {"blocks": blocks}}, parse_diag, resolution


# Helper functions for inference

def _has_grounding_tags(text: str) -> bool:
    """Check if text contains DeepSeek grounding tags."""
    return bool(text and "<|ref|>" in text and "<|det|>" in text)


def _clean_model_debug_output(text: str) -> str:
    """Remove model debug statistics from captured output.

    The model prints debug info like:
    ==================================================
    image size:  (1653, 2339)
    valid image tokens:  780
    output texts tokens (valid):  2785
    compression ratio:  3.57
    ==================================================

    This should be removed from the OCR content.
    """
    if not text:
        return text

    # Find the last occurrence of === separator pattern
    # The debug output is typically at the end after the last grounding tag
    lines = text.split('\n')

    # Find last line with === (at least 20 = signs)
    last_separator_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if '=' * 20 in lines[i]:
            last_separator_idx = i
            break

    # If we found a separator, check if there's another one before it (debug block pattern)
    if last_separator_idx > 0:
        # Look backward for the opening separator
        for i in range(last_separator_idx - 1, -1, -1):
            if '=' * 20 in lines[i]:
                # Found the opening separator - remove everything from here to end
                cleaned_lines = lines[:i]
                return '\n'.join(cleaned_lines).strip()

    return text


def _run_inference(
        model: Any,
        tokenizer: Any,
        image_path: Path,
        infer_config: Dict[str, Any],
        debug_dir: Optional[Path] = None,
) -> Tuple[str, float]:
    """Run model inference and capture stdout/stderr to buffers.

    Args:
        model: The model instance
        tokenizer: The tokenizer instance
        image_path: Path to input image
        infer_config: Inference configuration dictionary
        debug_dir: Optional directory to save raw stdout/stderr output

    Returns:
        (captured_text, inference_time_sec)
    """
    logger = logging.getLogger("deepseek_ocr")

    t0 = time.time()

    # Create StringIO buffers to capture output
    out_buf = StringIO()
    err_buf = StringIO()

    prompt = infer_config.get("prompt", PROMPT)
    config_for_infer = {k: v for k, v in infer_config.items() if k != "prompt"}

    # Log input image dimensions and model config
    try:
        img = Image.open(image_path)
        img_w, img_h = img.size
        logger.info(
            "[INFERENCE] Input: %s (%dx%d px) | Model config: base_size=%s, image_size=%s, prompt=%s",
            image_path.name,
            img_w,
            img_h,
            config_for_infer.get("base_size", "default"),
            config_for_infer.get("image_size", "default"),
            prompt
        )
    except Exception as e:
        logger.warning("[INFERENCE] Could not read input image dimensions: %s", e)

    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    exception_msg = None

    # Determine output path for inference artifacts
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        inference_output_path = str(debug_dir)
    else:
        inference_output_path = str(image_path.parent)

    try:
        # Redirect stdout/stderr to buffers
        sys.stdout = out_buf
        sys.stderr = err_buf

        # Run inference
        with torch.inference_mode():
            model.infer(
                tokenizer,
                prompt=prompt,
                image_file=str(image_path),
                output_path=inference_output_path,
                save_results=True,
                **config_for_infer,
            )
    except Exception as exc:
        exception_msg = f"{type(exc).__name__}: {exc}"
        # Restore stdout/stderr before logging
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.error("Inference exception: %s", exception_msg)
        import traceback
        traceback.print_exc()
    finally:
        # Always restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    infer_time = time.time() - t0

    # Get captured text from buffers
    captured = ""
    if out_buf.getvalue().strip():
        captured = out_buf.getvalue().strip()
        logger.debug("Captured from stdout: %d chars (raw)", len(captured))
    elif err_buf.getvalue().strip():
        captured = err_buf.getvalue().strip()
        logger.debug("Captured from stderr: %d chars (raw)", len(captured))
    else:
        logger.warning("No output captured from buffers")
        if exception_msg:
            captured = f"[INFERENCE_EXCEPTION] {exception_msg}"

    # Clean model debug output (statistics) from captured text
    if captured and not captured.startswith("[INFERENCE_EXCEPTION]"):
        captured_cleaned = _clean_model_debug_output(captured)
        if len(captured_cleaned) < len(captured):
            logger.debug("Removed %d chars of model debug output", len(captured) - len(captured_cleaned))
        captured = captured_cleaned

    # Save to debug file if requested (save the raw uncleaned version for debugging)
    if debug_dir:
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            # Save raw uncleaned output
            raw_output_path = debug_dir / "raw_output.txt"
            raw_with_debug = out_buf.getvalue().strip() or err_buf.getvalue().strip() or ""
            if raw_with_debug:
                raw_output_path.write_text(raw_with_debug, encoding="utf-8")
                logger.debug("Saved raw output (with debug stats) to: %s", raw_output_path)
        except Exception as e:
            logger.warning("Failed to save raw output: %s", e)

    return captured, infer_time


def run_page_ocr(
        model: Any,
        tokenizer: Any,
        image_path: Path,
        infer_config: Dict[str, Any],
        logger: logging.Logger,
        page_id: str = "",
        store_raw: bool = False,
        debug_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], float, str, List[int]]:
    """Run OCR on a single rendered page image.

    Single-pass inference only: crop_mode fallback has been removed.

    Args:
        store_raw: If True, store raw detection metadata for debugging
        debug_dir: Optional directory to save raw stdout/stderr output

    Returns:
        (page_ocr, inference_time_sec, attempt_used, resolution)
    """

    logger.info("[%s] Inference", page_id)
    captured, infer_time = _run_inference(model, tokenizer, image_path, infer_config, debug_dir)

    logger.debug("[%s] Attempt captured %d chars", page_id, len(captured))
    if captured:
        preview = captured[:200] if len(captured) > 200 else captured
        logger.debug("[%s] Output preview: %r", page_id, preview)
    else:
        logger.warning("[%s] Inference produced empty output", page_id)

    if _has_grounding_tags(captured):
        logger.info("[%s] Inference succeeded", page_id)
        page_ocr, _, resolution = parse_grounded_stdout(
            captured,
            image_path,
            logger,
            page_id=page_id,
            store_raw=store_raw,
        )
        return page_ocr, infer_time, "base", resolution

    logger.warning("[%s] No grounding tags detected", page_id)
    img = Image.open(image_path)
    if image_path.suffix.lower() == ".png" and image_path.name.startswith("page_") and img.mode != "RGB":
        img = img.convert("RGB")
    resolution = [img.size[0], img.size[1]]
    return {"ocr": {"blocks": []}}, infer_time, "none", resolution






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
        inference_config: Optional[Dict[str, Any]] = None,
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

    infer_config = inference_config if inference_config is not None else INFERENCE_CONFIG

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmp_dir = Path(tmpdir_str)

        if input_type == "pdf":
            pdf_doc = fitz.open(input_path)
            total_pages = len(pdf_doc)
            pdf_doc.close()

            selected_indices = parse_page_selection(pages, page_range, total_pages)

            page_image_paths = render_pdf_to_images(
                input_path, cfg.dpi, tmp_dir, logger, page_indices=selected_indices
            )
        else:
            # For direct image input, apply rotation detection if enabled
            img = Image.open(input_path)
            detected_rotation = detect_page_rotation(img, logger)

            if detected_rotation > 0:
                # If we rotate, we need an RGB-like image for saving/consistency.
                # Only convert if required by the underlying mode.
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")

                logger.info(
                    "[IMAGE] Rotation correction: detected=%d° → rotating image",
                    detected_rotation
                )
                img = img.rotate(-detected_rotation, expand=True, resample=Image.Resampling.BICUBIC)
                # Save rotated image to temp directory
                rotated_path = tmp_dir / f"{input_path.stem}_rotated{input_path.suffix}"
                img.save(rotated_path)
                page_image_paths = [rotated_path]
                logger.info(
                    "[IMAGE] Rotated image saved: %s (new size: %dx%d)",
                    rotated_path.name, img.width, img.height
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

            try:
                # Create debug directory for this page
                page_debug_dir = selection_dir / "debug" / f"page_{page_index:04d}"

                page_ocr, infer_time, attempt_used, resolution = run_page_ocr(
                    model=model,
                    tokenizer=tokenizer,
                    image_path=page_path,
                    infer_config=infer_config,
                    logger=logger,
                    page_id=page_id,
                    store_raw=cfg.store_raw_metadata,
                    debug_dir=page_debug_dir,
                )


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
                    img = Image.open(page_path)
                    # Keep raw mode for direct-image inputs; PDF inputs already recorded via render path.
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

    # Legacy document.json is no longer written - only per-page spec files

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
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "mps", "cpu"], help="Device selection")
    parser.add_argument("--pages", type=int, nargs="+", help="Page indices to process (1-based, e.g., 1 5 10)")
    parser.add_argument("--page-range", type=str, help="Inclusive page range (1-based, e.g., 1-10)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Console log level")
    parser.add_argument("--base-size", type=int, default=None, help="Override base_size (e.g., 512, 768, 1024, 1280)")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Override image_size (e.g., 384, 512, 640, 768, 896)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Maximum tokens to generate (default: model default)")
    parser.add_argument(
        "--raw",
        action="store_true",
        default=False,
        help="Store raw detection metadata (raw_label, raw_det, etc.) for debugging bbox parsing issues",
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

    _, resolved_device = resolve_device(args.device, logger)

    cfg = Config(
        dpi=args.dpi,
        out_dir=args.output_dir,
        log_level=args.log_level,
        store_raw_metadata=args.raw,
    )
    ds_cfg = DeepSeekConfig(device=resolved_device)

    input_path = Path(args.input).expanduser().resolve()
    try:
        inputs = gather_inputs(input_path)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 2
    if not inputs:
        logger.error("No valid PDF or image files found")
        return 2

    model, tokenizer = build_model(cfg.model, resolved_device, logger)

    # Start with INFERENCE_CONFIG as the base default
    inference_config = INFERENCE_CONFIG.copy()


    if args.base_size:
        inference_config["base_size"] = args.base_size
        logger.info("base_size overridden to %d", args.base_size)
    if args.image_size:
        inference_config["image_size"] = args.image_size
        logger.info("image_size overridden to %d", args.image_size)
    if args.max_new_tokens:
        inference_config["max_new_tokens"] = args.max_new_tokens
        logger.info("max_new_tokens set to %d", args.max_new_tokens)


    logger.info(
        "Config: base_size=%s, image_size=%s",
        inference_config.get("base_size"),
        inference_config.get("image_size"),
    )

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
        )

        selection_dir = doc_root / selection_label
        out_path = selection_dir

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

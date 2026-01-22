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
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

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
    "text": "paragraph",
    "title": "paragraph",
    "table": "table",
    "image": "image",
    "diagram": "diagram",
}

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Prefer a pixel-area cap rather than forcing a small max dimension.
# ~10–12 MP keeps tables readable without exploding runtime/memory.
MAX_PIXEL_AREA = 12_000_000

# Keep DPI sane; increase selectively only when needed.
DEFAULT_DPI = 200

DEFAULT_PROMPT_EXTRACT_ALL = (
    "<image>\n"
    "<|grounding|>Extract the full document content in reading order. "
    "1) Output grounded blocks using <|ref|>...</|ref|> and <|det|>...</|det|>. "
    "2) Do not omit any text. "
    "3) For tables: output as HTML <table>...</table> inside the corresponding block. "
    "4) Preserve numbers, units, and punctuation exactly. No paraphrasing."
)

PROMPT = DEFAULT_PROMPT_EXTRACT_ALL

# Mode-specific inference configurations
INFERENCE_MODES = {
    "extract_all": {
        "prompt": DEFAULT_PROMPT_EXTRACT_ALL,
        "base_size": 1024,
        "image_size": 768,
        "crop_mode": False,
    },
    "document": {
        "prompt": "<image>\n<|grounding|>Convert the document to markdown. Do not omit any rows or columns from tables. Preserve full table structure.",
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": False,
    },
    "chart": {
        "prompt": "<image>\n<|grounding|>Parse all charts and tables. Extract data as HTML tables.",
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": False,
    },
    "dense_text": {
        "prompt": "<image>\n<|grounding|>Free OCR.",
        "base_size": 768,
        "image_size": 512,
        "crop_mode": False,
    },
    "high_detail": {
        "prompt": "<image>\n<|grounding|>Convert the document to markdown. Do not omit any rows or columns from tables. Preserve full table structure.",
        "base_size": 1024,
        "image_size": 768,
        "crop_mode": False,
    },
    "multilingual": {
        "prompt": "<image>\n<|grounding|>Extract text. Preserve all languages and structure.",
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": False,
    },
    "experimental_tiny": {
        "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
        "base_size": 512,
        "image_size": 384,
        "crop_mode": False,
    },
    "experimental_large": {
        "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
        "base_size": 1280,
        "image_size": 896,
        "crop_mode": False,
    },
    "experimental_simple_prompt": {
        "prompt": "<image>\n<|grounding|>Extract all text.",
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": False,
    },
    "experimental_detailed_prompt": {
        "prompt": "<image>\n<|grounding|>Carefully extract all text from this document. Pay attention to tables, headers, and formatting. Do not skip any content.",
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": False,
    },
    "experimental_base_768_img_640": {
        "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
        "base_size": 768,
        "image_size": 640,
        "crop_mode": False,
    },
    "experimental_base_1024_img_512": {
        "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
        "base_size": 1024,
        "image_size": 512,
        "crop_mode": False,
    },
    "experimental_base_896_img_768": {
        "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
        "base_size": 896,
        "image_size": 768,
        "crop_mode": False,
    },
}

INFERENCE_CONFIG = {
    "base_size": 1024,
    "image_size": 768,
    "crop_mode": False,
    "test_compress": False,
    # Deterministic generation / stable outputs
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    # Prevent truncation on long pages (override via CLI if needed)
    "max_new_tokens": 8000,
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


# -------------------------
# Perf / debug log helpers
# -------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _fmt_bytes(num: Optional[int]) -> str:
    if num is None:
        return "n/a"
    try:
        n = float(num)
    except Exception:
        return str(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"


def _cuda_mem_snapshot() -> Dict[str, Any]:
    """Return a small CUDA memory snapshot (safe on CPU-only)."""
    if not torch.cuda.is_available():
        return {"cuda": False}
    try:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return {
            "cuda": True,
            "device": idx,
            "name": props.name,
            "total": int(props.total_memory),
            "allocated": int(torch.cuda.memory_allocated(idx)),
            "reserved": int(torch.cuda.memory_reserved(idx)),
            "max_allocated": int(torch.cuda.max_memory_allocated(idx)),
            "max_reserved": int(torch.cuda.max_memory_reserved(idx)),
        }
    except Exception as exc:
        return {"cuda": True, "error": f"{type(exc).__name__}: {exc}"}


class _PerfTimer:
    def __init__(self, logger: logging.Logger, scope: str, page_id: str = "", **fields: Any):
        self.logger = logger
        self.scope = scope
        self.page_id = page_id
        self.fields = fields
        self.t0 = time.time()
        self.cuda_before = _cuda_mem_snapshot()

    def __enter__(self):
        prefix = f"[{self.page_id}] " if self.page_id else ""
        self.logger.debug("%s[PERF][START] %s fields=%s cuda_before=%s", prefix, self.scope, self.fields,
                          _summarize_cuda(self.cuda_before))
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        cuda_after = _cuda_mem_snapshot()
        prefix = f"[{self.page_id}] " if self.page_id else ""
        base = {
            "scope": self.scope,
            "ms": int(dt * 1000),
            "fields": self.fields,
            "cuda_before": _summarize_cuda(self.cuda_before),
            "cuda_after": _summarize_cuda(cuda_after),
        }
        if exc_type is not None:
            base["error"] = f"{exc_type.__name__}: {exc}"
            self.logger.error("%s[PERF][END][ERROR] %s", prefix, base)
        else:
            # Emit INFO if it looks slow; otherwise DEBUG.
            if dt >= 5.0:
                self.logger.info("%s[PERF][END][SLOW] %s", prefix, base)
            else:
                self.logger.debug("%s[PERF][END] %s", prefix, base)
        return False


def _summarize_cuda(snap: Dict[str, Any]) -> Dict[str, Any]:
    if not snap.get("cuda"):
        return {"cuda": False}
    if "error" in snap:
        return {"cuda": True, "error": snap["error"]}
    return {
        "cuda": True,
        "device": snap.get("device"),
        "name": snap.get("name"),
        "allocated": _fmt_bytes(snap.get("allocated")),
        "reserved": _fmt_bytes(snap.get("reserved")),
        "max_allocated": _fmt_bytes(snap.get("max_allocated")),
        "max_reserved": _fmt_bytes(snap.get("max_reserved")),
        "total": _fmt_bytes(snap.get("total")),
    }


def _image_meta(path: Path) -> Dict[str, Any]:
    """Read minimal image metadata; avoids loading full pixel tensor."""
    try:
        with Image.open(path) as im:
            return {
                "path": str(path),
                "format": im.format,
                "mode": im.mode,
                "size": [im.size[0], im.size[1]],
            }
    except Exception as exc:
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}


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


def render_pdf_to_images(
        pdf_path: Path,
        dpi: int,
        tmp_dir: Path,
        logger: logging.Logger,
        page_indices: Optional[List[int]] = None,
) -> List[Path]:
    """Render selected PDF pages to images.

    Key behaviors:
      - Honors PDF page rotation (very common for landscape scans)
      - Avoids double-rotating: some PDFs have rotation metadata but content is already upright
      - Caps by pixel area (~12MP) to preserve table readability

    Strategy:
      - If rotation is 0/180: just render normally with that rotation.
      - If rotation is 90/270: render two candidates (apply rotation vs. ignore rotation)
        and pick the candidate whose aspect ratio matches the *expected upright view*.
        This prevents the regression where portrait pages become landscape.
    """

    def _render(page_obj: fitz.Page, rot: int) -> fitz.Pixmap:
        mat = fitz.Matrix(zoom, zoom).prerotate(rot)
        return page_obj.get_pixmap(matrix=mat, alpha=False)

    doc = fitz.open(pdf_path)
    image_paths: List[Path] = []

    zoom = dpi / 72.0

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

        pixmap: Optional[fitz.Pixmap] = None

        try:
            rotation = int(page.rotation or 0) % 360
        except Exception:
            rotation = 0

        chosen_rot = rotation
        chosen_reason = "apply_pdf_rotation"

        # For 90/270, PDF rotation is often correct… but sometimes the content is already upright.
        # We'll render both and pick the one that yields portrait-like output (more OCR-friendly).
        if rotation in (90, 270):
            with _PerfTimer(logger, "pdf.get_pixmap.candidates", page_id=f"P{page_index + 1:03d}", dpi=dpi, pdf_rotation=rotation, zoom=zoom):
                pix_apply = _render(page, rotation)
                pix_ignore = _render(page, 0)

            # Heuristic: prefer portrait (h>=w). Most documents/table pages are portrait.
            # If both are portrait/landscape, prefer the one closer to portrait.
            apply_portrait = pix_apply.height >= pix_apply.width
            ignore_portrait = pix_ignore.height >= pix_ignore.width

            if apply_portrait and not ignore_portrait:
                chosen_pix = pix_apply
                chosen_rot = rotation
                chosen_reason = "rotation_fixed_sideways"
            elif ignore_portrait and not apply_portrait:
                chosen_pix = pix_ignore
                chosen_rot = 0
                chosen_reason = "rotation_metadata_ignored_double_rotate"
            else:
                # Tie-breaker: choose the candidate with larger height/width ratio
                ratio_apply = pix_apply.height / max(1, pix_apply.width)
                ratio_ignore = pix_ignore.height / max(1, pix_ignore.width)
                if ratio_apply >= ratio_ignore:
                    chosen_pix = pix_apply
                    chosen_rot = rotation
                    chosen_reason = "tie_breaker_apply"
                else:
                    chosen_pix = pix_ignore
                    chosen_rot = 0
                    chosen_reason = "tie_breaker_ignore"

            pixmap = chosen_pix

            logger.info(
                "[RENDER][P%03d] pdf_rot=%d candidate_apply=%dx%d candidate_ignore=%dx%d chosen_rot=%d reason=%s",
                page_index + 1,
                rotation,
                pix_apply.width,
                pix_apply.height,
                pix_ignore.width,
                pix_ignore.height,
                chosen_rot,
                chosen_reason,
            )

        else:
            # 0 or 180: just apply rotation.
            with _PerfTimer(
                    logger,
                    "pdf.get_pixmap",
                    page_id=f"P{page_index + 1:03d}",
                    dpi=dpi,
                    rotation=rotation,
                    zoom=zoom,
            ):
                pixmap = _render(page, rotation)

        if pixmap is None:
            raise RuntimeError("Failed to render PDF page to pixmap")

        mode = "RGB" if pixmap.n < 4 else "RGBA"
        with _PerfTimer(
                logger,
                "pixmap.to_pil",
                page_id=f"P{page_index + 1:03d}",
                pix_w=pixmap.width,
                pix_h=pixmap.height,
                pix_n=pixmap.n,
                mode=mode,
        ):
            img = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
            if img.mode != "RGB":
                img = img.convert("RGB")

        # Pixel-area cap (~12MP) instead of hard max dimension.
        w, h = img.size
        area = int(w * h)
        if area > MAX_PIXEL_AREA:
            scale = (MAX_PIXEL_AREA / float(area)) ** 0.5
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            resample = getattr(Image, "LANCZOS", None) or getattr(Image.Resampling, "LANCZOS", 1)
            with _PerfTimer(
                    logger,
                    "pil.resize.pixel_cap",
                    page_id=f"P{page_index + 1:03d}",
                    original=[w, h],
                    new=list(new_size),
                    original_area=area,
                    cap=MAX_PIXEL_AREA,
                    scale=round(scale, 4),
            ):
                img = img.resize(new_size, resample)
            logger.info(
                "[RENDER][P%03d] pixel-cap resize %dx%d (%d px) -> %dx%d (%d px)",
                page_index + 1,
                w,
                h,
                area,
                new_size[0],
                new_size[1],
                int(new_size[0] * new_size[1]),
            )

        img_path = tmp_dir / f"page_{page_index}.png"
        with _PerfTimer(logger, "pil.save", page_id=f"P{page_index + 1:03d}", out=str(img_path)):
            img.save(img_path)
        image_paths.append(img_path)

        elapsed_sec = time.time() - t0
        logger.info(
            "[RENDER][P%03d] dpi=%d pix=%dx%d pdf_rot=%s applied_rot=%s time=%.3fs",
            page_index + 1,
            dpi,
            pixmap.width,
            pixmap.height,
            getattr(page, "rotation", None),
            chosen_rot,
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
    """Convert detection coordinates to normalized bbox [x, y, width, height] in 0-1 range."""
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

    norm_x = px1 / img_w if img_w > 0 else 0.0
    norm_y = py1 / img_h if img_h > 0 else 0.0
    norm_w = (px2 - px1) / img_w if img_w > 0 else 0.0
    norm_h = (py2 - py1) / img_h if img_h > 0 else 0.0

    return [
        max(0.0, min(1.0, norm_x)),
        max(0.0, min(1.0, norm_y)),
        max(0.0, min(1.0, norm_w)),
        max(0.0, min(1.0, norm_h)),
    ]


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


def build_spec_page_payload(
        document_id: str,
        page_id: str,
        resolution: List[int],
        blocks: List[Dict[str, Any]],
        warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a per-page spec payload."""
    payload = {
        "document_id": document_id,
        "page_id": page_id,
        "resolution": resolution,
        "ocr": {"blocks": blocks},
    }

    if warnings:
        payload["warnings"] = warnings

    return payload


def parse_grounded_stdout(
        captured: str,
        image_path: Path,
        logger: logging.Logger,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[int]]:
    """Parse grounded model output.

    Returns:
        (legacy_page_ocr, parse_diagnostics, resolution)
    """
    with _PerfTimer(logger, "parse.open_image", page_id=image_path.stem, image=image_path.name,
                    captured_chars=len(captured)):
        img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    resolution = [img_w, img_h]

    with _PerfTimer(logger, "parse.find_refs", page_id=image_path.stem, img_w=img_w, img_h=img_h):
        matches = list(REF_RE.finditer(captured))

    blocks: List[Dict[str, Any]] = []
    bbox_parse_failures = 0

    with _PerfTimer(logger, "parse.blocks", page_id=image_path.stem, match_count=len(matches)):
        for i, match in enumerate(matches):
            label = match.group("label").strip()
            det_text = match.group("det").strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(captured)
            content = captured[start:end].strip()

            try:
                det_coords = _parse_det(det_text)
                bbox = _det_to_bbox_normalized(det_coords, img_w, img_h)
            except Exception as exc:
                bbox = [0.0, 0.0, 0.0, 0.0]
                bbox_parse_failures += 1
                logger.warning(
                    "bbox parse failed for block %d on %s: %s: %s",
                    len(blocks),
                    image_path.name,
                    type(exc).__name__,
                    exc,
                )

            block_type = LABEL_TO_TYPE.get(label, "paragraph")

            if block_type == "table":
                table_data = try_parse_markdown_table(content)
                parsed_payload = {"data": table_data, "text": normalize_text(content)}
            elif block_type == "image":
                parsed_payload = {"data": None, "text": ""}
            else:
                parsed_payload = {"data": None, "text": normalize_text(content)}

            legacy_block = {
                "id": f"blk_{len(blocks) + 1}",
                "type": block_type,
                "order": len(blocks),
                "bbox": bbox,
                "extraction_response": content,
                "extraction_response_parsed": parsed_payload,
            }
            blocks.append(legacy_block)

    parse_diag = {
        "img_w": img_w,
        "img_h": img_h,
        "match_count": len(matches),
        "block_count": len(blocks),
        "bbox_parse_failures": bbox_parse_failures,
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
) -> Tuple[str, float]:
    """Run model inference and capture output.

    Returns:
        (captured_text, inference_time_sec)
    """
    logger = logging.getLogger("deepseek_ocr")
    t0 = time.time()
    out_buf = StringIO()
    err_buf = StringIO()

    prompt = infer_config.get("prompt", PROMPT)
    config_for_infer = {k: v for k, v in infer_config.items() if k != "prompt"}

    # Very verbose config dump (masked prompt preview)
    meta = {
        "image": _image_meta(image_path),
        "prompt_preview": (prompt[:120] + "…") if isinstance(prompt, str) and len(prompt) > 120 else prompt,
        "infer_kwargs": config_for_infer,
    }
    logger.info("[INFER][BEGIN] %s", meta)

    res = None
    exception_msg = None
    with _PerfTimer(logger, "hf.model.infer", page_id=image_path.stem, **{"image": str(image_path)}):
        with torch.inference_mode():
            try:
                with redirect_stdout(out_buf), redirect_stderr(err_buf):
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

    infer_time = time.time() - t0

    captured = ""
    if isinstance(res, str) and res.strip():
        captured = res.strip()
    elif out_buf.getvalue().strip():
        captured = out_buf.getvalue().strip()
    elif err_buf.getvalue().strip():
        captured = err_buf.getvalue().strip()

    # Capture sizes and short tails to debug hangs / huge outputs
    stdout_val = out_buf.getvalue()
    stderr_val = err_buf.getvalue()
    logger.info(
        "[INFER][END] image=%s time=%.3fs res_type=%s captured_chars=%d stdout_chars=%d stderr_chars=%d exception=%s",
        image_path.name,
        infer_time,
        type(res).__name__ if res is not None else None,
        len(captured),
        len(stdout_val),
        len(stderr_val),
        bool(exception_msg),
    )

    if logger.isEnabledFor(logging.DEBUG):
        if stdout_val.strip():
            logger.debug("[INFER][STDOUT][TAIL] %r", stdout_val[-2000:])
        if stderr_val.strip():
            logger.debug("[INFER][STDERR][TAIL] %r", stderr_val[-2000:])

    if exception_msg:
        logger.error("Inference exception: %s", exception_msg)
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
) -> Tuple[Dict[str, Any], float, str, List[int]]:
    """Run OCR on a single page image.

    Returns:
        (page_ocr, inference_time_sec, attempt_used, resolution)
    """
    if backend == Backend.OLLAMA:
        logger.info("[%s] Inference (ollama)", page_id)
        prompt = infer_config.get("prompt", DEFAULT_PROMPT_EXTRACT_ALL)
        captured, infer_time = _run_ollama_inference(image_path, logger, prompt=prompt)
        if _has_grounding_tags(captured):
            page_ocr, _, resolution = parse_grounded_stdout(captured, image_path, logger)
            return page_ocr, infer_time, "ollama", resolution
        logger.warning("[%s] No grounding tags detected", page_id)
        img = Image.open(image_path).convert("RGB")
        resolution = [img.size[0], img.size[1]]
        return {"ocr": {"blocks": []}}, infer_time, "none", resolution

    logger.info("[%s] Inference (base)", page_id)
    captured, time_base = _run_inference(model, tokenizer, image_path, infer_config)

    logger.debug("[%s] Base attempt captured %d chars", page_id, len(captured))
    if captured:
        preview = captured[:200] if len(captured) > 200 else captured
        logger.debug("[%s] Base output preview: %r", page_id, preview)
    else:
        logger.warning("[%s] Base attempt produced empty output", page_id)

    if _has_grounding_tags(captured):
        logger.info("[%s] Inference succeeded (base)", page_id)
        page_ocr, _, resolution = parse_grounded_stdout(captured, image_path, logger)
        return page_ocr, time_base, "base", resolution

    if not disable_fallbacks:
        logger.info("[%s] Inference (crop_mode fallback)", page_id)
        crop_config = dict(infer_config)
        crop_config["crop_mode"] = True
        captured_crop, time_crop = _run_inference(model, tokenizer, image_path, crop_config)

        logger.debug("[%s] Crop attempt captured %d chars", page_id, len(captured_crop))
        if captured_crop:
            preview_crop = captured_crop[:200] if len(captured_crop) > 200 else captured_crop
            logger.debug("[%s] Crop output preview: %r", page_id, preview_crop)
        else:
            logger.warning("[%s] Crop attempt produced empty output", page_id)

        if _has_grounding_tags(captured_crop):
            logger.warning("[%s] Base attempt failed; crop_mode fallback succeeded", page_id)
            page_ocr, _, resolution = parse_grounded_stdout(captured_crop, image_path, logger)
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
        norm_x, norm_y, norm_w, norm_h = bbox
        x = int(norm_x * img_w)
        y = int(norm_y * img_h)
        w = int(norm_w * img_w)
        h = int(norm_h * img_h)
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest_path)


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).convert("RGB").save(dst)


def gather_inputs(input_arg: Path) -> List[Path]:
    """Gather input files from a file or directory (PDFs and supported images)."""
    if input_arg.is_file():
        return [input_arg]
    if input_arg.is_dir():
        candidates: List[Path] = []
        for path in sorted(input_arg.rglob("*")):
            if path.is_file() and (is_pdf(path) or is_image(path)):
                candidates.append(path)
        return candidates
    raise FileNotFoundError(f"Input not found: {input_arg}")


def parse_page_selection(pages: Optional[List[int]], page_range: Optional[str], total_pages: int) -> List[int]:
    """Parse page selection flags into 0-based indices."""

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
    selection_dir = output_root / selection_label
    selection_dir.mkdir(parents=True, exist_ok=True)

    infer_config = inference_config if inference_config is not None else INFERENCE_CONFIG

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmp_dir = Path(tmpdir_str)

        if input_type == "pdf":
            pdf_doc = fitz.open(input_path)
            total_pages = len(pdf_doc)
            pdf_doc.close()

            selected_indices = parse_page_selection(pages, page_range, total_pages)
            with _PerfTimer(logger, "pdf.render_selected", page_id=input_path.stem, pages=len(selected_indices),
                            dpi=cfg.dpi):
                page_image_paths = render_pdf_to_images(
                    input_path, cfg.dpi, tmp_dir, logger, page_indices=selected_indices
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
            page_dir = selection_dir / f"page_{page_index + 1}"
            page_dir.mkdir(parents=True, exist_ok=True)

            page_id = f"{input_path.stem}|P{page_index + 1:03d}"

            logger.info("[%s] Page start: image=%s meta=%s", page_id, page_path.name, _image_meta(page_path))

            if cfg.debug_keep_renders:
                with _PerfTimer(logger, "page.persist_render", page_id=page_id):
                    try:
                        _copy_image(page_path, page_dir / "render.png")
                    except Exception as exc:
                        logger.warning("[%s] failed to persist render.png: %s: %s", page_id, type(exc).__name__, exc)

            try:
                with _PerfTimer(logger, "page.run_page_ocr", page_id=page_id, backend=str(cfg.backend)):
                    page_ocr, infer_time, attempt_used, resolution = run_page_ocr(
                        model=model,
                        tokenizer=tokenizer,
                        image_path=page_path,
                        infer_config=infer_config,
                        logger=logger,
                        page_id=page_id,
                        disable_fallbacks=cfg.disable_fallbacks,
                        backend=cfg.backend,
                    )

                if cfg.save_bbox_overlay:
                    with _PerfTimer(logger, "page.save_bbox_overlay", page_id=page_id):
                        try:
                            save_bbox_image(page_path, page_ocr.get("ocr", {}).get("blocks", []),
                                            page_dir / "page_bbox.png")
                        except Exception as exc:
                            logger.warning("[%s] failed to save bbox overlay: %s: %s", page_id, type(exc).__name__, exc)

                document_id = input_path.stem
                page_id_str = str(page_index)

                spec_blocks: List[Dict[str, Any]] = []
                for blk in page_ocr.get("ocr", {}).get("blocks", []):
                    spec_blocks.append(
                        {
                            "type": blk["type"],
                            "bbox": blk["bbox"],
                            "extraction_origin": blk.get("extraction_origin", "deepseek-ocr"),
                            "extraction_response": blk["extraction_response"],
                            "extraction_response_parsed": blk["extraction_response_parsed"],
                        }
                    )

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

                with _PerfTimer(logger, "page.write_json", page_id=page_id,
                                out=str(selection_dir / f"page_{page_index + 1}.json")):
                    page_json_path = selection_dir / f"page_{page_index + 1}.json"
                    page_json_path.write_text(json.dumps(spec_payload, ensure_ascii=False, indent=2), encoding="utf-8")

                logger.info(
                    "[%s] Page done: attempt=%s blocks=%d infer_time=%.3fs resolution=%s",
                    page_id,
                    attempt_used,
                    len(spec_blocks),
                    infer_time,
                    resolution,
                )

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
                with _PerfTimer(logger, "page.write_json.error", page_id=page_id,
                                out=str(selection_dir / f"page_{page_index + 1}.json")):
                    page_json_path = selection_dir / f"page_{page_index + 1}.json"
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
    parser.add_argument("--eval-output-dir", default="./evaluation_output",
                        help="Directory for evaluation run summaries (default: ./evaluation_output)")
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
        "--modes",
        nargs="+",
        choices=list(INFERENCE_MODES.keys()),
        default=None,
        help="Run multiple modes (space-separated). Overrides --mode when used.",
    )
    parser.add_argument(
        "--run-modes-per-page",
        action="store_true",
        default=False,
        help="If set, run each selected mode for each selected page and save into per-mode output folders.",
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
    return parser.parse_args(argv)


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
        eval_out_dir=args.eval_output_dir,
        log_level=args.log_level,
        debug_keep_renders=bool(args.keep_renders),
        save_bbox_overlay=not args.no_bbox_overlay,
        disable_fallbacks=not args.enable_fallbacks if not args.disable_fallbacks else args.disable_fallbacks,
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

    # Determine which modes we will run.
    # If --run-modes-per-page is set and --modes is not provided, run ALL modes.
    if args.modes:
        modes_to_run = args.modes
    elif args.run_modes_per_page:
        modes_to_run = list(INFERENCE_MODES.keys())
    else:
        modes_to_run = [args.mode]

    if args.run_modes_per_page:
        logger.info("Modes-per-page enabled; modes=%s", modes_to_run)

    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = Path(cfg.eval_out_dir).expanduser().resolve()

    run_items: List[Dict[str, Any]] = []
    grand_total_time = 0.0
    total_pages = 0

    run_start = time.time()
    for input_file in inputs:
        base_selection_label = "all_pages"
        if args.pages:
            base_selection_label = "pages_" + "-".join(map(str, args.pages))
        elif args.page_range:
            base_selection_label = f"pages_{args.page_range}"

        for mode_name in modes_to_run:
            inference_config = INFERENCE_MODES.get(mode_name, INFERENCE_MODES["extract_all"]).copy()

            # Allow global overrides even in multi-mode runs
            if args.prompt is not None:
                inference_config["prompt"] = args.prompt
                logger.info("Prompt: custom (--prompt) [mode=%s]", mode_name)
            else:
                logger.info("Prompt: mode default (%s)", mode_name)

            if args.test_compress:
                inference_config["test_compress"] = True
                logger.info("test_compress enabled [mode=%s]", mode_name)

            if args.base_size:
                inference_config["base_size"] = args.base_size
                logger.info("base_size overridden to %d [mode=%s]", args.base_size, mode_name)
            if args.image_size:
                inference_config["image_size"] = args.image_size
                logger.info("image_size overridden to %d [mode=%s]", args.image_size, mode_name)
            if args.max_new_tokens:
                inference_config["max_new_tokens"] = args.max_new_tokens
                logger.info("max_new_tokens set to %d [mode=%s]", args.max_new_tokens, mode_name)

            if cfg.disable_fallbacks:
                logger.info("Crop mode fallback: disabled")
            else:
                logger.info("Crop mode fallback: enabled")

            logger.info(
                "Mode: %s (base_size=%s, image_size=%s)",
                mode_name,
                inference_config.get("base_size"),
                inference_config.get("image_size"),
            )

            if args.legacy_document_json:
                logger.info("Legacy document.json output: enabled")
            else:
                logger.info("Legacy document.json output: disabled")

            # Separate results by mode when running a grid.
            selection_label = base_selection_label
            if args.run_modes_per_page and len(modes_to_run) > 1:
                selection_label = f"{base_selection_label}__mode_{mode_name}"

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
                    "mode": mode_name,
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

#!/usr/bin/env python3
"""DeepSeek-OCR pipeline: extract structured text/tables from PDFs and images."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import warnings
import logging
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# Import common utilities from the new common module
from lsextractor.pipeline.common import (
    parse_page_selection,
    build_spec_page_payload,
    detect_page_rotation,
    resize_image_if_needed,
    classify_header_footer_heuristic,
    render_pdf_pages,
)

# Import DeepSeek engine-specific modules
from lsextractor.pipeline.engines.deepseek import (
    DEFAULT_INFERENCE_CONFIG,
    merge_inference_config,
    parse_grounded_stdout,
    _has_grounding_tags,
    _clean_model_debug_output,
)

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    pytesseract = None

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
    class _DummyColor:
        def __getattr__(self, name):
            return ""
    Fore = _DummyColor()
    Style = _DummyColor()
    COLORS_AVAILABLE = False


ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


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

@dataclass
class Config:
    """Configuration for DeepSeek-OCR pipeline."""
    model: str = "deepseek-ai/DeepSeek-OCR"
    dpi: int = 200
    out_dir: str = "./out_json/deepseek_ocr"
    eval_out_dir: str = "./evaluation_output"
    log_level: str = "INFO"
    store_raw_metadata: bool = False
    max_input_size: Optional[int] = None

@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek model loading."""
    device: str = "auto"


def is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"

def is_image(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS

def resolve_device(device_arg: str, logger: logging.Logger) -> Tuple[str, str]:
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

def build_model(model_name: str, device: str, logger: logging.Logger) -> Tuple[Any, Any]:
    """Load DeepSeek-OCR model and tokenizer."""
    logger.info("Loading model %s on %s", model_name, device)
    logger.info("Note: first run will download ~6GB model files from HuggingFace")

    gpu_major, gpu_minor = None, None
    if device == "cuda" and torch.cuda.is_available():
        gpu_major, gpu_minor = torch.cuda.get_device_capability(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)} ({gpu_mem_gb:.1f} GB) - Compute Capability: SM{gpu_major}{gpu_minor}")

    if device == "cuda" and torch.cuda.is_available():
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

    use_flash_attn = False
    flash_attn_working = False
    if device == "cuda" and gpu_major is not None and gpu_major >= 8:
        try:
            import flash_attn
            from flash_attn.flash_attn_interface import flash_attn_func
            flash_attn_working = True
            use_flash_attn = True
            logger.info("FlashAttention-2 detected and working")
        except (ImportError, OSError, AttributeError) as e:
            if "ImportError" in str(type(e).__name__) and "flash_attn" not in str(e).lower():
                logger.warning(f"Flash attention installed but has import errors: {e}")
                logger.warning("Will use eager attention. To fix: pip uninstall flash-attn && pip install flash-attn --no-build-isolation")
            elif "flash_attn" not in str(e).lower():
                logger.warning(f"Flash attention check failed: {e}")
            else:
                logger.info("Flash attention not installed; using eager attention")
                logger.info("For faster inference (optional): pip install flash-attn --no-build-isolation")
            use_flash_attn = False
            flash_attn_working = False

    tokenizer_kwargs = {"trust_remote_code": True}
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "use_safetensors": True,
    }

    if use_flash_attn and flash_attn_working:
        model_kwargs["_attn_implementation"] = "flash_attention_2"
        logger.info("Will use FlashAttention-2 for model")
    else:
        model_kwargs["_attn_implementation"] = "eager"
        logger.info("Will use eager attention (standard PyTorch)")

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    logger.info("Tokenizer loaded, loading model (this may take a minute)...")
    model = AutoModel.from_pretrained(model_name, **model_kwargs)

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


def render_pdf_to_images(pdf_path: Path, dpi: int, tmp_dir: Path, logger: logging.Logger,
                         page_indices: Optional[List[int]] = None, max_input_size: Optional[int] = None) -> List[Path]:
    """Render selected PDF pages to images (wrapper for common.render_pdf_pages)."""
    return render_pdf_pages(
        pdf_path=pdf_path,
        dpi=dpi,
        out_dir=tmp_dir,
        logger=logger,
        page_indices=page_indices,
        max_input_size=max_input_size,
    )


def _run_inference(model: Any, tokenizer: Any, image_path: Path, infer_config: Dict[str, Any],
                  debug_dir: Optional[Path] = None) -> Tuple[str, float]:
    """Run model inference and capture stdout/stderr to buffers."""
    logger = logging.getLogger("deepseek_ocr")
    t0 = time.time()
    out_buf = StringIO()
    err_buf = StringIO()
    prompt = infer_config.get("prompt", DEFAULT_INFERENCE_CONFIG["prompt"])
    config_for_infer = {k: v for k, v in infer_config.items() if k != "prompt"}

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

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    exception_msg = None

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        inference_output_path = str(debug_dir)
    else:
        inference_output_path = str(image_path.parent)

    try:
        sys.stdout = out_buf
        sys.stderr = err_buf
        with torch.inference_mode():
            model.infer(tokenizer, prompt=prompt, image_file=str(image_path),
                       output_path=inference_output_path, save_results=True, **config_for_infer)
    except Exception as exc:
        exception_msg = f"{type(exc).__name__}: {exc}"
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.error("Inference exception: %s", exception_msg)
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    infer_time = time.time() - t0

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

    if captured and not captured.startswith("[INFERENCE_EXCEPTION]"):
        captured_cleaned = _clean_model_debug_output(captured)
        if len(captured_cleaned) < len(captured):
            logger.debug("Removed %d chars of model debug output", len(captured) - len(captured_cleaned))
        captured = captured_cleaned

    if debug_dir:
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            raw_output_path = debug_dir / "raw_output.txt"
            raw_with_debug = out_buf.getvalue().strip() or err_buf.getvalue().strip() or ""
            if raw_with_debug:
                raw_output_path.write_text(raw_with_debug, encoding="utf-8")
                logger.debug("Saved raw output (with debug stats) to: %s", raw_output_path)
        except Exception as e:
            logger.warning("Failed to save raw output: %s", e)

    return captured, infer_time

def run_page_ocr(model: Any, tokenizer: Any, image_path: Path, infer_config: Dict[str, Any],
                logger: logging.Logger, page_id: str = "", store_raw: bool = False,
                debug_dir: Optional[Path] = None) -> Tuple[Dict[str, Any], float, str, List[int]]:
    """Run OCR on a single rendered page image."""
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
    """Process one input document and write per-page outputs."""
    warnings_list: List[str] = []
    input_type = "pdf" if is_pdf(input_path) else "image"

    if selection_label and selection_label != "all_pages":
        selection_dir = output_root / selection_label
    else:
        selection_dir = output_root
    selection_dir.mkdir(parents=True, exist_ok=True)

    infer_config = inference_config if inference_config is not None else DEFAULT_INFERENCE_CONFIG

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmp_dir = Path(tmpdir_str)

        if input_type == "pdf":
            pdf_doc = fitz.open(input_path)
            total_pages = len(pdf_doc)
            pdf_doc.close()

            selected_indices = parse_page_selection(pages, page_range, total_pages)

            page_image_paths = render_pdf_to_images(
                input_path, cfg.dpi, tmp_dir, logger,
                page_indices=selected_indices,
                max_input_size=cfg.max_input_size
            )
        else:
            img = Image.open(input_path)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            if cfg.max_input_size:
                img, was_resized = resize_image_if_needed(img, cfg.max_input_size, logger)
                if was_resized:
                    logger.info("[IMAGE] Resized to %dx%d (max_input_size=%d)",
                               img.size[0], img.size[1], cfg.max_input_size)

            detected_rotation = detect_page_rotation(img, logger)

            if detected_rotation > 0:
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")

                logger.info("[IMAGE] Rotation correction: detected=%d° → rotating image", detected_rotation)
                img = img.rotate(-detected_rotation, expand=True, resample=Image.Resampling.BICUBIC)

                if cfg.max_input_size:
                    img, was_resized = resize_image_if_needed(img, cfg.max_input_size, logger)
                    if was_resized:
                        logger.info("[IMAGE] Resized rotated image to %dx%d (max_input_size=%d)",
                                   img.size[0], img.size[1], cfg.max_input_size)

                rotated_path = tmp_dir / f"{input_path.stem}_rotated{input_path.suffix}"
                img.save(rotated_path)
                page_image_paths = [rotated_path]
                logger.info("[IMAGE] Rotated image saved: %s (new size: %dx%d)",
                           rotated_path.name, img.width, img.height)
            else:
                if cfg.max_input_size and img.size != Image.open(input_path).size:
                    resized_path = tmp_dir / f"{input_path.stem}_resized{input_path.suffix}"
                    img.save(resized_path)
                    page_image_paths = [resized_path]
                    logger.info("[IMAGE] Resized image saved: %s (size: %dx%d)",
                               resized_path.name, img.width, img.height)
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
            page_id = f"{input_path.stem}|P{page_index:04d}"

            try:
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

                    if cfg.store_raw_metadata:
                        for key in ["raw_label", "raw_det", "raw_det_parsed", "raw_bbox_parse_failed",
                                   "raw_match_index", "raw_is_corrupted"]:
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
    parser.add_argument("--max-input-size", type=int, default=None,
                        help="Max pixels for longer edge before inference (e.g., 2560). Resizes maintaining aspect ratio.")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Maximum tokens to generate (default: model default)")
    parser.add_argument(
        "--raw",
        action="store_true",
        default=False,
        help="Store raw detection metadata (raw_label, raw_det, etc.) for debugging bbox parsing issues",
    )
    return parser.parse_args(argv)



def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logger = _setup_logger(args.log_level)

    _, resolved_device = resolve_device(args.device, logger)

    cfg = Config(
        dpi=args.dpi,
        out_dir=args.output_dir,
        log_level=args.log_level,
        store_raw_metadata=args.raw,
        max_input_size=args.max_input_size,
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

    inference_config = merge_inference_config(None)

    if args.base_size:
        inference_config["base_size"] = args.base_size
        logger.info("base_size overridden to %d", args.base_size)
    if args.image_size:
        inference_config["image_size"] = args.image_size
        logger.info("image_size overridden to %d", args.image_size)
    if args.max_new_tokens:
        inference_config["max_new_tokens"] = args.max_new_tokens
        logger.info("max_new_tokens set to %d", args.max_new_tokens)

    if cfg.max_input_size:
        logger.info("max_input_size set to %d px (longer edge)", cfg.max_input_size)

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
#!/usr/bin/env python3
"""
dots.ocr Pipeline - Document OCR with Layout Extraction

Server-ready implementation (rednote-hilab/dots.ocr). Requires 10GB+ GPU VRAM.
Features: Prompt fallback, pixel-cap downscaling, unified output schema.
Install: pip install -e ".[dotsocr]"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import fitz  # PyMuPDF
from PIL import Image

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    class _DummyColor:
        def __getattr__(self, name):
            return ""
    Fore = _DummyColor()
    Style = _DummyColor()
    COLORS_AVAILABLE = False
    def colorama_init(**kwargs):
        pass


MODEL_ID = "rednote-hilab/dots.ocr"
MAX_PIXEL_AREA = 11_289_600

PROMPT_MODES = {
    "layout_all_en": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.

IMPORTANT: The JSON must have the exact top-level format: {"elements":[ ... ]} (do not output a bare JSON array).
IMPORTANT: For every element except category="Picture", include a non-empty "text" field. For Table, "text" must be valid HTML (<table>...</table>).""",

    "ocr": """Extract all text from this document image in reading order. Return plain text only.""",

    "layout_only_en": """Detect all layout blocks in this document image including bounding boxes and categories.
Categories: Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title

IMPORTANT: The JSON must have the exact top-level format: {"elements":[ ... ]} (do not output a bare JSON array).
Return ONLY the JSON object.""",
}

CATEGORY_TO_TYPE = {
    "Title": "header",
    "Section-header": "header",
    "Text": "paragraph",
    "List-item": "paragraph",
    "Caption": "paragraph",
    "Footnote": "paragraph",
    "Table": "table",
    "Formula": "paragraph",
    "Picture": "image",
    "Page-header": "page-header",
    "Page-footer": "page-footer",
}

RUNAWAY_PATTERNS = [
    re.compile(r"\.{10,}"),
    re.compile(r"_{10,}"),
    re.compile(r"-{10,}"),
    re.compile(r"\s{50,}"),
]

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
    logger = logging.getLogger("dots_ocr")
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
    """Configuration for dots.ocr pipeline."""
    model: str = MODEL_ID
    dpi: int = 200
    out_dir: str = "./out_json/dots_ocr"
    eval_out_dir: str = "./evaluation_output"
    prompt_mode: str = "layout_all_en"
    max_new_tokens: int = 24000
    fallback_enabled: bool = True
    save_rendered_images: bool = False
    save_bbox_overlay: bool = True
    log_level: str = "INFO"
    device: str = "auto"
    dtype: str = "bfloat16"


@dataclass
class DotsOcrConfig:
    """Configuration for dots.ocr model loading."""
    model_path: Optional[str] = None
    device: str = "auto"
    dtype: str = "bfloat16"
    trust_remote_code: bool = True



@dataclass
class InferResult:
    raw_text: str
    parsed_json: Optional[Dict[str, Any]]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    def infer_page(self, image: Image.Image, prompt_mode: str, max_new_tokens: int) -> InferResult:
        ...


class TransformersBackend:
    """HuggingFace Transformers backend for dots.ocr inference."""

    def __init__(self, cfg: DotsOcrConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self._model = None
        self._processor = None
        self._load_model()

    @staticmethod
    def _flash_attn2_available() -> bool:
        try:
            import flash_attn
            return True
        except Exception:
            return False

    def _load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_path = self.cfg.model_path or MODEL_ID
        self.logger.info(f"Loading dots.ocr model from {model_path}...")
        self.logger.info("Note: First run will download ~4-8GB model files from HuggingFace")

        if model_path == MODEL_ID and not self.cfg.model_path:
            self.logger.warning("Model name 'dots.ocr' contains dot - downloading to local directory")
            from pathlib import Path

            local_model_dir = Path.home() / ".cache" / "dots_ocr_renamed"
            local_model_dir.mkdir(parents=True, exist_ok=True)

            if not (local_model_dir / "config.json").exists():
                self.logger.info(f"Downloading model to {local_model_dir}...")
                try:
                    from huggingface_hub import snapshot_download

                    snapshot_download(repo_id=MODEL_ID, local_dir=str(local_model_dir))
                    self.logger.info("Model downloaded successfully")
                except ImportError:
                    self.logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
                    raise
            else:
                self.logger.info(f"Using cached model from {local_model_dir}")

            model_path = str(local_model_dir)

        device_map = "auto" if self.cfg.device == "auto" else self.cfg.device

        gpu_major, gpu_minor = None, None
        if torch.cuda.is_available():
            gpu_major, gpu_minor = torch.cuda.get_device_capability(0)

        if self.cfg.dtype == "bfloat16":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
                self.logger.info("BF16 supported - using bfloat16 for optimal performance")
            elif torch.cuda.is_available():
                torch_dtype = torch.float16
                self.logger.warning("BF16 not supported on this GPU (e.g., T4). Using float16 instead")
            else:
                torch_dtype = torch.float32
                self.logger.warning("CUDA not available. Using float32 on CPU")
        elif self.cfg.dtype == "float16":
            torch_dtype = torch.float16
            self.logger.info("Using explicit float16 (user requested)")
        elif self.cfg.dtype == "float32":
            torch_dtype = torch.float32
            self.logger.info("Using explicit float32 (user requested)")
        elif self.cfg.dtype == "auto":
            self.logger.warning("dtype='auto' not recommended (causes mixed precision). Auto-selecting based on hardware")
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
                self.logger.info("BF16 supported - using bfloat16")
            elif torch.cuda.is_available():
                torch_dtype = torch.float16
                self.logger.info("Using float16")
            else:
                torch_dtype = torch.float32
                self.logger.info("Using float32 on CPU")
        else:
            torch_dtype = torch.float32
            self.logger.warning(f"Unknown dtype '{self.cfg.dtype}', defaulting to float32")

        self.logger.info(f"Using dtype: {torch_dtype}")

        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            self.logger.info(f"CUDA available: {torch.cuda.get_device_name(0)} ({gpu_mem_gb:.1f} GB) - Compute Capability: SM{gpu_major}{gpu_minor}")

            if gpu_mem_gb < 8:
                self.logger.error(
                    f"CRITICAL: GPU memory is {gpu_mem_gb:.1f} GB. dots.ocr requires ~10GB+ VRAM for inference"
                )
                self.logger.error("Inference will fail with OutOfMemoryError. Use --device cpu (very slow but will work)")

                if device_map != "cpu":
                    raise RuntimeError(
                        f"Insufficient GPU memory ({gpu_mem_gb:.1f} GB). "
                        f"dots.ocr requires ~10GB+ VRAM. Use --device cpu or test in Colab"
                    )
            elif gpu_mem_gb < 12:
                self.logger.warning(
                    f"GPU memory is {gpu_mem_gb:.1f} GB. dots.ocr may encounter OOM errors. Recommended: 12GB+ VRAM"
                )
        else:
            self.logger.warning("CUDA not available; using CPU (inference will be very slow)")

        use_flash_attn = torch.cuda.is_available() and gpu_major >= 8 and self._flash_attn2_available()

        if torch.cuda.is_available() and gpu_major >= 8:
            if use_flash_attn:
                self.logger.info("Using FlashAttention-2 (installed + Ampere+ GPU detected)")
            else:
                self.logger.warning("Ampere+ GPU detected but flash_attn not installed; using default attention")
                self.logger.warning("For faster inference, install: pip install flash-attn --no-build-isolation")
        elif torch.cuda.is_available():
            self.logger.info(f"Using default attention (SM{gpu_major}{gpu_minor} < SM80, FlashAttention-2 not supported)")

        try:
            self.logger.info(f"Loading processor from {model_path}...")
            self._processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=self.cfg.trust_remote_code,
                use_fast=False,
                local_files_only=True if self.cfg.model_path else False,
            )

            model_kwargs = {
                "trust_remote_code": self.cfg.trust_remote_code,
                "device_map": device_map,
                "torch_dtype": torch_dtype,
                "local_files_only": True if self.cfg.model_path else False,
            }
            if use_flash_attn:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            self.logger.info(f"Loading model from {model_path}...")
            self.logger.info(f"Model loading with dtype={torch_dtype}, device_map={device_map}")
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path, **model_kwargs
            ).eval()

        except Exception as exc:
            if "ModuleNotFoundError" in str(type(exc).__name__) and "dots" in str(exc):
                self.logger.error("=" * 80)
                self.logger.error("MODEL LOADING FAILED: The model name 'dots.ocr' contains a dot")
                self.logger.error("Python treats the dot as a module separator, causing import errors")
                self.logger.error("")
                self.logger.error("SOLUTION: Model should be auto-downloaded and renamed")
                self.logger.error("If this persists, manually:")
                self.logger.error("  1. Download: huggingface-cli download rednote-hilab/dots.ocr")
                self.logger.error("  2. Rename folder: dots.ocr -> dots_ocr")
                self.logger.error("  3. Use: --model-path /path/to/dots_ocr")
                self.logger.error("=" * 80)
            raise

        self.logger.info("Model loaded successfully")

        if torch_dtype not in ["auto", None]:
            param_dtypes = {p.dtype for p in self._model.parameters()}
            if len(param_dtypes) > 1:
                self.logger.warning(f"Model has mixed precision parameters: {param_dtypes}")
                self.logger.warning(f"Converting entire model to uniform dtype: {torch_dtype}")

            self._model = self._model.to(dtype=torch_dtype)

            final_dtypes = {p.dtype for p in self._model.parameters()}
            if len(final_dtypes) == 1:
                self.logger.info(f"Model unified to dtype: {next(iter(final_dtypes))}")
            else:
                self.logger.error(f"Warning: Model still has mixed dtypes after conversion: {final_dtypes}")

    def _cast_floating_inputs_to_model_dtype(self, inputs):
        """Cast floating-point tensors in inputs to match model's dtype."""
        import torch
        model_dtype = next(self._model.parameters()).dtype

        for k in list(inputs.keys()):
            v = inputs[k]
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point and v.dtype != model_dtype:
                inputs[k] = v.to(dtype=model_dtype)

        return inputs

    def infer_page(self, image: Image.Image, prompt_mode: str, max_new_tokens: int) -> InferResult:
        import torch
        from qwen_vl_utils import process_vision_info

        t0 = time.time()
        warnings_list = []

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_MODES.get(prompt_mode, PROMPT_MODES["layout_all_en"])},
            ],
        }]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self._model.device)

        inputs = self._cast_floating_inputs_to_model_dtype(inputs)

        with torch.inference_mode():
            try:
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                )
            except Exception as exc:
                self.logger.error(f"Generation failed: {type(exc).__name__}: {exc}")
                return InferResult(
                    raw_text="", parsed_json=None,
                    diagnostics={
                        "prompt_mode": prompt_mode, "time_sec": time.time() - t0,
                        "output_tokens": 0, "warnings": [f"generation_error: {type(exc).__name__}: {exc}"],
                    }
                )

        input_len = inputs.input_ids.shape[1]
        generated_ids_trimmed = [out_ids[input_len:] for out_ids in generated_ids]

        decoder = getattr(self._processor, "tokenizer", self._processor)
        output_text = decoder.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        output_text = output_text.replace("<|endofassistant|>", "").strip()
        output_text = output_text.replace("<|im_end|>", "").strip()

        output_tokens = len(generated_ids_trimmed[0])
        infer_time = time.time() - t0

        for pattern in RUNAWAY_PATTERNS:
            if pattern.search(output_text):
                warnings_list.append(f"runaway_pattern_detected: {pattern.pattern}")
                self.logger.warning(f"Runaway pattern detected: {pattern.pattern}")

        parsed_json, parse_warnings = parse_model_output(output_text, self.logger)
        warnings_list.extend(parse_warnings)

        return InferResult(
            raw_text=output_text, parsed_json=parsed_json,
            diagnostics={"prompt_mode": prompt_mode, "time_sec": round(infer_time, 4),
                        "output_tokens": output_tokens, "warnings": warnings_list}
        )


class PipelineBackend:
    """HuggingFace Transformers pipeline backend for dots.ocr."""

    def __init__(self, cfg: DotsOcrConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self._pipe = None
        self._load_pipeline()

    def _load_pipeline(self):
        import torch
        from transformers import pipeline

        model_path = self.cfg.model_path or MODEL_ID
        self.logger.info(f"Loading dots.ocr pipeline from {model_path}...")
        self.logger.info("Note: Pipeline API is simpler but has less control over model loading")

        gpu_major, gpu_minor = None, None
        torch_dtype = self.cfg.dtype
        if torch.cuda.is_available():
            gpu_major, gpu_minor = torch.cuda.get_device_capability(0)
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            self.logger.info(f"CUDA available: {torch.cuda.get_device_name(0)} ({gpu_mem_gb:.1f} GB) - Compute Capability: SM{gpu_major}{gpu_minor}")

            if torch_dtype == "bfloat16" and gpu_major < 8:
                self.logger.warning("GPU likely lacks bf16 support (e.g., T4). Switching dtype to float16")
                torch_dtype = "float16"
        else:
            self.logger.warning("CUDA not available; using CPU (inference will be very slow)")

        pipe_kwargs = {
            "model": model_path,
            "trust_remote_code": self.cfg.trust_remote_code,
            "torch_dtype": torch_dtype,
        }

        device_map = "auto" if self.cfg.device == "auto" else self.cfg.device
        if device_map != "auto":
            pipe_kwargs["device"] = device_map

        try:
            self._pipe = pipeline("image-text-to-text", **pipe_kwargs)
            self.logger.info("Pipeline loaded successfully")
        except Exception as exc:
            self.logger.error(f"Pipeline loading failed: {type(exc).__name__}: {exc}")
            if "trust_remote_code" in str(exc):
                self.logger.error("Try running with trust_remote_code=True")
            raise

    def infer_page(self, image: Image.Image, prompt_mode: str, max_new_tokens: int) -> InferResult:
        import time

        t0 = time.time()
        warnings_list = []

        prompt_text = PROMPT_MODES.get(prompt_mode, PROMPT_MODES["layout_all_en"])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        try:
            result = self._pipe(text=messages, max_new_tokens=max_new_tokens)

            if isinstance(result, list) and len(result) > 0:
                output_text = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                output_text = result.get("generated_text", "")
            else:
                output_text = str(result)

            infer_time = time.time() - t0

            for pattern in RUNAWAY_PATTERNS:
                if pattern.search(output_text):
                    warnings_list.append(f"runaway_pattern_detected: {pattern.pattern}")
                    self.logger.warning(f"Runaway pattern detected: {pattern.pattern}")

            parsed_json, parse_warnings = parse_model_output(output_text, self.logger)
            warnings_list.extend(parse_warnings)

            return InferResult(
                raw_text=output_text,
                parsed_json=parsed_json,
                diagnostics={
                    "prompt_mode": prompt_mode,
                    "time_sec": round(infer_time, 4),
                    "output_tokens": len(output_text.split()),
                    "warnings": warnings_list
                }
            )

        except Exception as exc:
            self.logger.error(f"Pipeline inference failed: {type(exc).__name__}: {exc}")
            return InferResult(
                raw_text="",
                parsed_json=None,
                diagnostics={
                    "prompt_mode": prompt_mode,
                    "time_sec": time.time() - t0,
                    "output_tokens": 0,
                    "warnings": [f"generation_error: {type(exc).__name__}: {exc}"]
                }
            )


def parse_model_output(raw_text: str, logger: Optional[logging.Logger] = None) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Parse dots.ocr output into a dict."""
    warnings = []
    if not raw_text or not raw_text.strip():
        return None, ["empty_output"]

    text = raw_text.strip()

    text = text.replace("<|endofassistant|>", "").strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            warnings.append("wrapped_list_into_elements")
            parsed = {"elements": parsed}

        if isinstance(parsed, dict):
            validation_result = _validate_dots_schema(parsed, logger)
            if validation_result:
                warnings.extend(validation_result)
            return parsed, warnings

        warnings.append(f"unexpected_json_type:{type(parsed).__name__}")
        return None, warnings
    except Exception:
        pass

    extracted = _extract_first_balanced_json(text)
    if extracted is not None:
        try:
            parsed = json.loads(extracted)
            warnings.append("json_extracted_balanced")
            if isinstance(parsed, list):
                warnings.append("wrapped_list_into_elements")
                parsed = {"elements": parsed}

            if isinstance(parsed, dict):
                validation_result = _validate_dots_schema(parsed, logger)
                if validation_result:
                    warnings.extend(validation_result)
                return parsed, warnings
        except Exception:
            pass

    warnings.append("json_parse_failed")
    if logger:
        preview = text[:1200] + ("..." if len(text) > 1200 else "")
        logger.warning(f"JSON parse failed. Output preview:\n{preview}")
    return None, warnings


def _validate_dots_schema(parsed: Dict[str, Any], logger: Optional[logging.Logger] = None) -> List[str]:
    """Validate dots.ocr JSON schema."""
    warnings = []

    if "elements" not in parsed:
        warnings.append("schema_error:missing_elements_key")
        return warnings

    elements = parsed["elements"]
    if not isinstance(elements, list):
        warnings.append("schema_error:elements_not_list")
        return warnings

    allowed_categories = {
        'Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer',
        'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'
    }

    for idx, elem in enumerate(elements):
        if not isinstance(elem, dict):
            warnings.append(f"schema_error:element_{idx}_not_dict")
            continue

        if "bbox" not in elem:
            warnings.append(f"schema_error:element_{idx}_missing_bbox")
        elif not isinstance(elem["bbox"], list) or len(elem["bbox"]) != 4:
            warnings.append(f"schema_error:element_{idx}_invalid_bbox")

        if "category" not in elem:
            warnings.append(f"schema_error:element_{idx}_missing_category")
        elif elem["category"] not in allowed_categories:
            warnings.append(f"schema_warning:element_{idx}_unknown_category_{elem['category']}")

        category = elem.get("category", "")
        if category != "Picture" and "text" not in elem:
            warnings.append(f"schema_warning:element_{idx}_missing_text_for_{category}")
        elif category != "Picture" and not elem.get("text", "").strip():
            warnings.append(f"schema_warning:element_{idx}_empty_text_for_{category}")

    return warnings


def _extract_first_balanced_json(s: str) -> Optional[str]:
    """Extract first balanced {...} or [...] from string."""
    starts = [(s.find("{"), "{"), (s.find("["), "[")]
    starts = [(i, ch) for (i, ch) in starts if i != -1]
    if not starts:
        return None

    start_idx, start_ch = min(starts, key=lambda x: x[0])
    end_ch = "}" if start_ch == "{" else "]"

    depth = 0
    in_str = False
    esc = False

    for i in range(start_idx, len(s)):
        c = s[i]

        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue

        if c == '"':
            in_str = True
            continue

        if c == start_ch:
            depth += 1
        elif c == end_ch:
            depth -= 1
            if depth == 0:
                return s[start_idx:i + 1]

    return None


def run_page_ocr_with_fallback(
    backend: Backend, image: Image.Image, cfg: Config, logger: logging.Logger, page_id: str = ""
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Run OCR with prompt fallback chain: layout_all_en → layout_only_en."""
    fallback_chain = ["layout_all_en", "layout_only_en"] if cfg.fallback_enabled else [cfg.prompt_mode]
    all_diagnostics, total_time = [], 0.0

    for attempt_idx, prompt_mode in enumerate(fallback_chain, start=1):
        logger.info(f"[{page_id}] Attempt {attempt_idx}: prompt_mode={prompt_mode}")
        result = backend.infer_page(image, prompt_mode, cfg.max_new_tokens)
        total_time += result.diagnostics.get("time_sec", 0.0)
        all_diagnostics.append(result.diagnostics)

        if result.parsed_json:
            logger.info(f"[{page_id}] ✓ SUCCESS with {prompt_mode}")
            return result.parsed_json, {
                "attempts": all_diagnostics, "successful_prompt_mode": prompt_mode,
                "total_time_sec": round(total_time, 4),
            }
        else:
            logger.warning(f"[{page_id}] ✗ FAILED with {prompt_mode}: {result.diagnostics.get('warnings', [])}")

    logger.error(f"[{page_id}] ✗ All attempts FAILED")
    return None, {"attempts": all_diagnostics, "successful_prompt_mode": None, "total_time_sec": round(total_time, 4)}


def render_pdf_to_images(
    pdf_path: Path, dpi: int, tmp_dir: Path, logger: logging.Logger, page_indices: Optional[List[int]] = None
) -> List[Tuple[Path, Dict[str, Any]]]:
    """Render PDF pages to images with pixel-cap downscaling.

    Args:
        pdf_path: Input PDF path.
        dpi: Render DPI.
        tmp_dir: Output directory for rendered page images.
        logger: Logger instance.
        page_indices: Optional list of 0-based page indices to render.

    Returns:
        List of tuples (image_path, render_metadata).
    """
    doc = fitz.open(pdf_path)
    results, mat = [], fitz.Matrix(dpi / 72.0, dpi / 72.0)
    total_pages = len(doc)

    pages_to_render = (
        sorted({idx for idx in page_indices if 0 <= idx < total_pages}) if page_indices
        else list(range(total_pages))
    )
    logger.info(f"Rendering {len(pages_to_render)}/{total_pages} selected pages" if page_indices else f"Rendering all {total_pages} pages")

    for page_index in pages_to_render:
        t0 = time.time()
        pix = doc[page_index].get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB" if pix.n < 4 else "RGBA", (pix.width, pix.height), pix.samples)
        if img.mode != "RGB":
            img = img.convert("RGB")

        original_w, original_h = img.size
        downscaled = original_w * original_h > MAX_PIXEL_AREA

        if downscaled:
            scale_factor = (MAX_PIXEL_AREA / (original_w * original_h)) ** 0.5
            final_w, final_h = int(original_w * scale_factor), int(original_h * scale_factor)
            img = img.resize((final_w, final_h), Image.Resampling.LANCZOS)
            logger.info(f"[P{page_index+1:03d}] Downscaled {original_w}x{original_h} → {final_w}x{final_h}")
        else:
            final_w, final_h = original_w, original_h

        img_path = tmp_dir / f"page_{page_index}.png"
        img.save(img_path)

        results.append((img_path, {
            "page_index": page_index, "dpi": dpi,
            "original_size": [original_w, original_h], "final_size": [final_w, final_h],
            "downscaled": downscaled, "render_time_sec": round(time.time() - t0, 4),
        }))

        logger.info(f"[RENDER][P{page_index+1:03d}] dpi={dpi} size={final_w}x{final_h} downscaled={downscaled} time={time.time()-t0:.3f}s")

    doc.close()
    return results


def _parse_html_table_to_structure(html_text: str, logger: Optional[logging.Logger] = None) -> Optional[Dict[str, Any]]:
    """Parse HTML table into structured format with headers and rows.

    Returns:
        Dict with "headers" and "rows" keys, or None if parsing fails.
    """
    try:
        from html.parser import HTMLParser

        class TableParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.in_thead = False
                self.in_tbody = False
                self.in_tr = False
                self.in_th = False
                self.in_td = False
                self.headers = []
                self.rows = []
                self.current_row = []
                self.current_cell = []

            def handle_starttag(self, tag, attrs):
                if tag == "thead":
                    self.in_thead = True
                elif tag == "tbody":
                    self.in_tbody = True
                elif tag == "tr":
                    self.in_tr = True
                    self.current_row = []
                elif tag == "th":
                    self.in_th = True
                    self.current_cell = []
                elif tag == "td":
                    self.in_td = True
                    self.current_cell = []

            def handle_endtag(self, tag):
                if tag == "thead":
                    self.in_thead = False
                elif tag == "tbody":
                    self.in_tbody = False
                elif tag == "tr":
                    self.in_tr = False
                    if self.in_thead or (not self.in_thead and not self.in_tbody):
                        # Row in <thead> or before <tbody> = header row
                        if self.current_row:
                            self.headers.append(self.current_row)
                    else:
                        # Row in <tbody> = data row
                        if self.current_row:
                            self.rows.append(self.current_row)
                    self.current_row = []
                elif tag == "th":
                    self.in_th = False
                    cell_text = "".join(self.current_cell).strip()
                    self.current_row.append(cell_text)
                    self.current_cell = []
                elif tag == "td":
                    self.in_td = False
                    cell_text = "".join(self.current_cell).strip()
                    self.current_row.append(cell_text)
                    self.current_cell = []

            def handle_data(self, data):
                if self.in_th or self.in_td:
                    self.current_cell.append(data)

        parser = TableParser()
        parser.feed(html_text)

        # Return structured data
        return {
            "headers": parser.headers if parser.headers else None,
            "rows": parser.rows if parser.rows else []
        }

    except Exception as exc:
        if logger:
            logger.warning(f"Failed to parse HTML table: {type(exc).__name__}: {exc}")
        return None


def normalize_dots_output(
    page_index: int, dots_json: Dict[str, Any], img_size: Tuple[int, int],
    logger: logging.Logger
) -> Dict[str, Any]:
    """Normalize dots.ocr JSON to unified block schema with normalized bboxes [x1, y1, x2, y2] in 0-1 range."""
    img_w, img_h = img_size
    elements = dots_json.get("elements", [])

    if not isinstance(elements, list):
        logger.warning(f"No 'elements' array found in dots.ocr output for page {page_index}")
        return {"ocr": {"blocks": []}}

    blocks = []
    for elem_idx, elem in enumerate(elements):
        if not isinstance(elem, dict):
            continue

        category = elem.get("category", "Text")
        block_type = CATEGORY_TO_TYPE.get(category, "paragraph")

        bbox_raw = elem.get("bbox", [0, 0, 0, 0])
        if not (isinstance(bbox_raw, list) and len(bbox_raw) == 4):
            bbox_raw = [0, 0, 0, 0]

        x1, y1, x2, y2 = bbox_raw

        px1 = max(0.0, min(float(x1), float(img_w)))
        py1 = max(0.0, min(float(y1), float(img_h)))
        px2 = max(0.0, min(float(x2), float(img_w)))
        py2 = max(0.0, min(float(y2), float(img_h)))

        norm_x1 = px1 / img_w if img_w > 0 else 0.0
        norm_y1 = py1 / img_h if img_h > 0 else 0.0
        norm_x2 = px2 / img_w if img_w > 0 else 0.0
        norm_y2 = py2 / img_h if img_h > 0 else 0.0

        bbox = [
            max(0.0, min(1.0, norm_x1)),
            max(0.0, min(1.0, norm_y1)),
            max(0.0, min(1.0, norm_x2)),
            max(0.0, min(1.0, norm_y2)),
        ]

        text = elem.get("text", "") if block_type != "image" else ""

        if block_type == "table" and text:
            table_data = _parse_html_table_to_structure(text, logger)
            parsed_payload = {
                "data": table_data,
                "text": text
            }
        elif block_type == "image":
            parsed_payload = {"data": None, "text": ""}
        else:
            parsed_payload = {"data": None, "text": text}

        blocks.append({
            "id": f"blk_{page_index}_{elem_idx}",
            "type": block_type,
            "order": elem_idx,
            "bbox": bbox,
            "extraction_origin": "dots-ocr",
            "extraction_response": text if text else json.dumps(elem),
            "extraction_response_parsed": parsed_payload,
        })

    return {"ocr": {"blocks": blocks}}


def is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"

def is_image(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS

def gather_inputs(input_arg: Path) -> List[Path]:
    if input_arg.is_file():
        return [input_arg]
    if input_arg.is_dir():
        return sorted([p for p in input_arg.rglob("*") if p.is_file() and (is_pdf(p) or is_image(p))])
    raise FileNotFoundError(f"Input not found: {input_arg}")

def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).convert("RGB").save(dst)


def save_bbox_image(page_img: Image.Image, blocks: List[Dict[str, Any]], dest_path: Path) -> None:
    """Draw normalized bounding boxes on page image and save."""
    from PIL import ImageDraw

    img = page_img.copy()
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)

    for blk in blocks:
        bbox = blk.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue

        norm_x1, norm_y1, norm_x2, norm_y2 = bbox
        x1 = int(norm_x1 * img_w)
        y1 = int(norm_y1 * img_h)
        x2 = int(norm_x2 * img_w)
        y2 = int(norm_y2 * img_h)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest_path)


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


def process_document(
    backend: Backend,
    input_path: Path,
    cfg: Config,
    output_root: Path,
    selection_label: str,
    logger: logging.Logger,
    pages: Optional[List[int]] = None,
    page_range: Optional[str] = None,
    legacy_document_json: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Process a single document (PDF or image).

    Args:
        backend: OCR backend instance.
        input_path: Input file path.
        cfg: Configuration.
        output_root: Output directory root.
        selection_label: Label for page selection.
        logger: Logger instance.
        pages: Optional list of 1-based page indices.
        page_range: Optional page range string.
        legacy_document_json: Whether to write legacy document.json.

    Returns:
        Tuple of (doc_json, summary_metrics).
    """
    warnings_list: List[str] = []
    input_type = "pdf" if is_pdf(input_path) else "image"
    selection_dir = output_root / selection_label
    selection_dir.mkdir(parents=True, exist_ok=True)

    page_image_paths: List[Tuple[Path, Dict[str, Any]]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        if input_type == "pdf":
            doc_temp = fitz.open(input_path)
            total_pages = len(doc_temp)
            doc_temp.close()

            selected_indices = parse_page_selection(pages, page_range, total_pages)

            page_image_paths = render_pdf_to_images(
                input_path, cfg.dpi, tmp_path, logger, page_indices=selected_indices
            )
        else:
            img = Image.open(input_path).convert("RGB")
            img_path = tmp_path / "page_0.png"
            img.save(img_path)
            render_meta = {
                "page_index": 0,
                "dpi": None,
                "original_size": list(img.size),
                "final_size": list(img.size),
                "downscaled": False,
                "render_time_sec": 0.0,
            }
            page_image_paths = [(img_path, render_meta)]
            selected_indices = [0]

        pages_out: List[Dict[str, Any]] = []
        page_timings: List[Dict[str, Any]] = []
        total_time = 0.0
        total_output_tokens = 0

        for img_path, render_meta in page_image_paths:
            page_index = render_meta["page_index"]
            page_dir = selection_dir / f"page_{page_index + 1}"
            page_dir.mkdir(parents=True, exist_ok=True)

            page_id = f"{input_path.stem}|P{page_index+1:03d}"

            if cfg.save_rendered_images:
                try:
                    _copy_image(img_path, page_dir / "render.png")
                except Exception as exc:
                    logger.warning(f"[{page_id}] Failed to save render.png: {exc}")

            try:
                page_img = Image.open(img_path).convert("RGB")
                img_size = page_img.size
                resolution = [img_size[0], img_size[1]]

                dots_json, ocr_diagnostics = run_page_ocr_with_fallback(
                    backend, page_img, cfg, logger, page_id
                )

                if dots_json:
                    page_ocr = normalize_dots_output(
                        page_index, dots_json, img_size, logger
                    )
                else:
                    page_ocr = {"ocr": {"blocks": []}}
                    warnings_list.append(f"Page {page_index}: OCR failed after all fallbacks")

                page_time = ocr_diagnostics.get("total_time_sec", 0.0)
                total_time += page_time

                attempts = ocr_diagnostics.get("attempts", [])
                page_output_tokens = attempts[-1].get("output_tokens", 0) if attempts else 0
                total_output_tokens += page_output_tokens

                if cfg.save_bbox_overlay:
                    try:
                        save_bbox_image(page_img, page_ocr.get("ocr", {}).get("blocks", []), page_dir / "page_bbox.png")
                    except Exception as exc:
                        logger.warning(f"[{page_id}] failed to save bbox overlay: {type(exc).__name__}: {exc}")

                spec_blocks: List[Dict[str, Any]] = []
                for blk in page_ocr.get("ocr", {}).get("blocks", []):
                    spec_blocks.append({
                        "type": blk["type"],
                        "bbox": blk["bbox"],
                        "extraction_origin": blk.get("extraction_origin", "dots-ocr"),
                        "extraction_response": blk["extraction_response"],
                        "extraction_response_parsed": blk["extraction_response_parsed"],
                    })

                page_warnings: List[str] = []
                attempts_data = ocr_diagnostics.get("attempts", [])
                if not attempts_data:
                    page_warnings.append("OCR extraction failed - no attempts recorded")
                elif len(attempts_data) > 1:
                    page_warnings.append(f"Initial attempt(s) failed, used fallback (attempts: {len(attempts_data)})")

                if len(spec_blocks) == 0 and attempts_data:
                    page_warnings.append("No blocks extracted from page")

                document_id = input_path.stem
                page_id_str = str(page_index)
                spec_payload = build_spec_page_payload(
                    document_id=document_id,
                    page_id=page_id_str,
                    resolution=resolution,
                    blocks=spec_blocks,
                    warnings=page_warnings if page_warnings else None,
                )
                page_json_path = selection_dir / f"page_{page_index + 1}.json"
                page_json_path.write_text(json.dumps(spec_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.debug(f"[{page_id}] wrote {page_json_path.name}")

                page_payload = {
                    "meta": {
                        "model": cfg.model,
                        "input_file": str(input_path),
                        "input_type": input_type,
                        "page_index": page_index,
                        "page_label": page_index + 1,
                        "page_id": page_id,
                        "render_meta": render_meta,
                    },
                    "page_ocr": page_ocr,
                    "metrics": {
                        "time_sec": round(page_time, 4),
                        "output_tokens": page_output_tokens,
                    },
                    "diagnostics": ocr_diagnostics,
                }

                pages_out.append(page_payload)

                page_timings.append({
                    "page_index": page_index,
                    "page_label": page_index + 1,
                    "time_sec": round(page_time, 4),
                    "output_tokens": page_output_tokens,
                    "blocks_extracted": len(spec_blocks),
                    "attempts": len(attempts_data),
                })

            except Exception as exc:
                error_msg = f"Page {page_index} failed: {type(exc).__name__}: {exc}"
                warnings_list.append(error_msg)
                logger.error(f"[{page_id}] PAGE FAILED: {exc}")

                try:
                    img = Image.open(img_path).convert("RGB")
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
                    "metrics": {"time_sec": None, "output_tokens": 0},
                    "diagnostics": {"error": error_msg},
                }
                pages_out.append(page_payload)

                page_timings.append({
                    "page_index": page_index,
                    "page_label": page_index + 1,
                    "time_sec": None,
                    "output_tokens": 0,
                    "blocks_extracted": 0,
                    "error": error_msg,
                })

    page_count = len(pages_out)
    summary = {
        "total_pages": page_count,
        "total_time_sec": round(total_time, 4),
        "total_output_tokens": total_output_tokens,
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
            "prompt_mode": cfg.prompt_mode,
            "fallback_enabled": cfg.fallback_enabled,
            "warnings": warnings_list,
            "output_dir": str(selection_dir),
        },
        "pages": pages_out,
        "metrics_summary": summary,
    }

    # Only write legacy document.json if requested
    if legacy_document_json:
        out_path = selection_dir / "document.json"
        out_path.write_text(json.dumps(doc_json, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Legacy document.json written to {out_path}")

    return doc_json, summary



def parse_page_selection(
    pages: Optional[List[int]],
    page_range: Optional[str],
    total_pages: int
) -> List[int]:
    """Parse page selection arguments to 0-based page indices."""
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


def write_run_summary(
    eval_dir: Path,
    run_items: List[Dict[str, Any]],
    total_time: float,
    total_pages: int,
    device: str,
    model: str,
) -> Path:
    """Write run summary to evaluation output directory."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    ts_filename = time.strftime("%Y%m%d_%H%M%S")
    ts_readable = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path = eval_dir / f"dots_ocr_run_{ts_filename}.json"
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
    parser = argparse.ArgumentParser(
        description="dots.ocr pipeline for document OCR with layout extraction"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a PDF, image, or folder of files"
    )
    parser.add_argument(
        "--output-dir", default="./out_json/dots_ocr",
        help="Output directory for JSON files"
    )
    parser.add_argument(
        "--eval-output-dir", default="./evaluation_output",
        help="Directory for evaluation run summaries (default: ./evaluation_output)"
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="DPI for PDF rendering (default: 200)"
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="Device selection (default: auto)"
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16)"
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Local model path (default: download from HuggingFace)"
    )
    parser.add_argument(
        "--prompt-mode", default="layout_all_en",
        choices=list(PROMPT_MODES.keys()),
        help="Prompt mode (default: layout_all_en)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=24000,
        help="Maximum new tokens to generate (default: 24000, official recommendation for layout extraction)"
    )
    parser.add_argument(
        "--disable-fallback", action="store_true",
        help="Disable prompt-mode fallback chain"
    )
    parser.add_argument(
        "--save-rendered-images", action="store_true",
        help="Save rendered page images for debugging"
    )
    parser.add_argument(
        "--no-bbox-overlay", action="store_true",
        help="Disable saving page_bbox.png with bounding boxes drawn (default: enabled)"
    )
    parser.add_argument(
        "--pages", type=int, nargs="+",
        help="Page indices to process (1-based, e.g., 1 5 10)"
    )
    parser.add_argument(
        "--page-range", type=str,
        help="Inclusive page range (1-based, e.g., 1-10)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level"
    )
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

    cfg = Config(
        dpi=args.dpi,
        out_dir=args.output_dir,
        eval_out_dir=args.eval_output_dir,
        prompt_mode=args.prompt_mode,
        max_new_tokens=args.max_new_tokens,
        fallback_enabled=not args.disable_fallback,
        save_rendered_images=args.save_rendered_images,
        save_bbox_overlay=not args.no_bbox_overlay,
        log_level=args.log_level,
        device=args.device,
        dtype=args.dtype,
    )

    backend_cfg = DotsOcrConfig(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
    )

    input_path = Path(args.input).expanduser().resolve()
    try:
        inputs = gather_inputs(input_path)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 2

    if not inputs:
        logger.error("No valid PDF or image files found")
        return 2

    backend = TransformersBackend(backend_cfg, logger)
    logger.info("Backend loaded. Starting OCR...")

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
    for inp in inputs:
        selection_label = "all_pages"
        if args.pages:
            selection_label = "pages_" + "-".join(map(str, args.pages))
        elif args.page_range:
            selection_label = f"pages_{args.page_range}"

        doc_root = out_dir / inp.stem
        doc_root.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing document: {inp.name} (selection={selection_label})")

        doc_json, summary = process_document(
            backend,
            inp,
            cfg,
            output_root=doc_root,
            selection_label=selection_label,
            logger=logger,
            pages=args.pages,
            page_range=args.page_range,
            legacy_document_json=args.legacy_document_json,
        )

        selection_dir = doc_root / selection_label
        out_path = selection_dir / "document.json" if args.legacy_document_json else selection_dir

        run_items.append({
            "input_file": str(inp),
            "output_file": str(out_path),
            "pages": summary["total_pages"],
            "total_time_sec": summary["total_time_sec"],
            "total_output_tokens": summary.get("total_output_tokens", 0),
            "page_timings": summary.get("page_timings", []),
        })

        grand_total_time += summary["total_time_sec"]
        total_pages += summary["total_pages"]

    run_total_time = time.time() - run_start

    summary_path = write_run_summary(
        eval_dir=eval_dir,
        run_items=run_items,
        total_time=grand_total_time,
        total_pages=total_pages,
        device=backend_cfg.device,
        model=cfg.model,
    )

    logger.info(f"Processed {len(inputs)} documents. JSON saved to {out_dir}")
    logger.info(f"Run summary saved to {summary_path}")
    logger.info(f"Elapsed wall time: {run_total_time:.2f}s (model load included)")

    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    sys.exit(main())
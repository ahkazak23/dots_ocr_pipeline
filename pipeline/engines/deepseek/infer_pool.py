#!/usr/bin/env python3
"""DeepSeek subprocess inference pool to isolate stdout/stderr and enable parallel processing."""

import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from lsextractor.pipeline.engines.deepseek.parse import _clean_model_debug_output

_MODEL = None
_TOKENIZER = None

logger = logging.getLogger(__name__)


def init_deepseek_worker(
    model_name: str,
    device: str,
    torch_dtype: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Initialize DeepSeek model and tokenizer in subprocess (runs once per worker)."""
    global _MODEL, _TOKENIZER

    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch_dtype = "float32"

    if tokenizer_kwargs is None:
        tokenizer_kwargs = {"trust_remote_code": True}

    if model_kwargs is None:
        model_kwargs = {
            "trust_remote_code": True,
            "use_safetensors": True,
            "_attn_implementation": "eager",
        }

    _TOKENIZER = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    _MODEL = AutoModel.from_pretrained(model_name, **model_kwargs)
    _MODEL = _MODEL.eval()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    target_dtype = dtype_map.get(torch_dtype, torch.float32)

    if device == "cuda":
        _MODEL = _MODEL.cuda().to(target_dtype)
    elif device == "mps":
        _MODEL = _MODEL.to("mps").to(target_dtype)
    else:
        _MODEL = _MODEL.to("cpu").to(torch.float32)


def infer_one_image(
    image_path: str,
    infer_config: Dict[str, Any],
    debug_dir: Optional[str],
) -> Tuple[str, float, str]:
    """Run inference on one image in subprocess, returning (captured_text, time_sec, exception_str)."""
    global _MODEL, _TOKENIZER

    if _MODEL is None or _TOKENIZER is None:
        return "[INFERENCE_EXCEPTION] Model not initialized", 0.0, "RuntimeError: Model not initialized"

    t0 = time.time()
    out_buf = StringIO()
    err_buf = StringIO()

    prompt = infer_config.get("prompt", "<image>\n<|grounding|>Convert the document to markdown.")
    config_for_infer = {k: v for k, v in infer_config.items() if k != "prompt"}

    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        inference_output_path = debug_dir
    else:
        inference_output_path = str(Path(image_path).parent)

    exception_str = ""
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        sys.stdout = out_buf
        sys.stderr = err_buf
        with torch.inference_mode():
            _MODEL.infer(
                _TOKENIZER,
                prompt=prompt,
                image_file=image_path,
                output_path=inference_output_path,
                save_results=True,
                **config_for_infer
            )
    except Exception as exc:
        exception_str = f"{type(exc).__name__}: {exc}"
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    infer_time = time.time() - t0

    captured = ""
    if out_buf.getvalue().strip():
        captured = out_buf.getvalue().strip()
    elif err_buf.getvalue().strip():
        captured = err_buf.getvalue().strip()

    if exception_str:
        if not captured:
            captured = f"[INFERENCE_EXCEPTION] {exception_str}"
        else:
            captured = f"[INFERENCE_EXCEPTION] {exception_str}\n{captured}"

    if captured and not captured.startswith("[INFERENCE_EXCEPTION]"):
        captured = _clean_model_debug_output(captured)

    return captured, infer_time, exception_str


class DeepSeekInferPool:
    """Process pool for DeepSeek inference with isolated stdout/stderr."""

    def __init__(
        self,
        model_name: str,
        device: str,
        torch_dtype: str = "float32",
        max_workers: int = 1,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize inference pool with model loading in subprocess workers."""
        self.model_name = model_name
        self.device = device

        if device == "cpu" and torch_dtype != "float32":
            logger.info(f"CPU mode: coercing torch_dtype from {torch_dtype} to float32")
            torch_dtype = "float32"

        self.torch_dtype = torch_dtype
        self.max_workers = max_workers
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs

        ctx = multiprocessing.get_context("spawn")
        self._pool = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=init_deepseek_worker,
            initargs=(model_name, device, torch_dtype, model_kwargs, tokenizer_kwargs),
        )

    def run(
        self,
        image_path: Path | str,
        infer_config: Dict[str, Any],
        debug_dir: Optional[Path | str] = None,
    ) -> Tuple[str, float, str]:
        """Run inference on one image, returning (captured_text, time_sec, exception_str)."""
        future = self._pool.submit(
            infer_one_image,
            str(image_path),
            infer_config,
            str(debug_dir) if debug_dir else None,
        )
        return future.result()

    def close(self) -> None:
        """Shutdown the pool and wait for workers to finish."""
        if self._pool:
            self._pool.shutdown(wait=True)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False





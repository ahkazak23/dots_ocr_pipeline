#!/usr/bin/env python3
"""DeepSeek OCR engine module."""

from .config import DEFAULT_INFERENCE_CONFIG, merge_inference_config
from .parse import parse_grounded_stdout, _has_grounding_tags, _clean_model_debug_output
from .infer_pool import DeepSeekInferPool, init_deepseek_worker, infer_one_image

__all__ = [
    "DEFAULT_INFERENCE_CONFIG",
    "merge_inference_config",
    "parse_grounded_stdout",
    "_has_grounding_tags",
    "_clean_model_debug_output",
    "DeepSeekInferPool",
    "init_deepseek_worker",
    "infer_one_image",
]


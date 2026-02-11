#!/usr/bin/env python3
"""DeepSeek OCR engine configuration."""

from typing import Any, Dict, Optional


DEFAULT_INFERENCE_CONFIG: Dict[str, Any] = {
    "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
    "base_size": 1024,
    "image_size": 640,
    "crop_mode": True,
    "test_compress": True,
}


def merge_inference_config(user_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge user config with defaults, returning a new dict with user overrides applied."""
    merged = DEFAULT_INFERENCE_CONFIG.copy()
    if user_cfg:
        merged.update(user_cfg)
    return merged


#!/usr/bin/env python3
"""OmniDocBench spec payload builder."""

from typing import Dict, Any, List, Optional


def build_spec_page_payload(
    document_id: str,
    page_id: str,
    resolution: List[int],
    blocks: List[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build OmniDocBench-compatible per-page spec payload with meta, ocr blocks, and optional warnings."""
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

    # Add top-level warnings if provided (for compatibility)
    if warnings:
        payload["warnings"] = warnings

    return payload



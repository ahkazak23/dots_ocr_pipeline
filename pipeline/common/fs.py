#!/usr/bin/env python3
"""File system utilities for deterministic output paths and atomic JSON writes."""

import json
import tempfile
from pathlib import Path
from typing import Optional, List, Any, Dict


def build_selection_label(
    pages: Optional[List[int]],
    page_range: Optional[str]
) -> str:
    """Build deterministic selection label: 'pages_1-3', 'pages_1-2-7', or 'all_pages'."""
    if pages:
        return "pages_" + "-".join(map(str, pages))
    elif page_range:
        return f"pages_{page_range}"
    else:
        return "all_pages"


def build_page_output_dir(
    output_root: str,
    engine_name: str,
    document_id: str,
    selection_label: str
) -> str:
    """Build deterministic output directory path: output_root/engine_name/document_id/selection_label."""
    base_dir = Path(output_root) / engine_name / document_id
    if selection_label == "all_pages":
        return str(base_dir)
    else:
        return str(base_dir / selection_label)


def build_page_json_path(output_dir: str, page_index: int) -> str:
    """Build path for per-page JSON: output_dir/page_XXXX.json (4-digit zero-padded)."""
    return str(Path(output_dir) / f"page_{page_index:04d}.json")


def build_page_debug_dir(output_dir: str, page_index: int) -> str:
    """Build path for per-page debug directory: output_dir/debug/page_XXXX."""
    return str(Path(output_dir) / "debug" / f"page_{page_index:04d}")


def ensure_dir(path: str) -> None:
    """Ensure directory exists, creating parent directories as needed."""
    Path(path).mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: str, obj: Any) -> None:
    """Atomically write JSON object to file using temp file + rename."""
    target = Path(path)
    ensure_dir(str(target.parent))

    # Write to temp file in same directory for atomic rename
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=str(target.parent),
        delete=False,
        suffix='.tmp'
    ) as tmp:
        json.dump(obj, tmp, ensure_ascii=False, indent=2)
        tmp_path = Path(tmp.name)

    # Atomic rename
    tmp_path.replace(target)


def read_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    """Read JSON file if it exists, return None otherwise."""
    file_path = Path(path)
    if not file_path.exists():
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def validate_page_payload_minimal(payload: Dict[str, Any]) -> bool:
    """Validate that page payload has required keys: 'meta', 'ocr', and 'ocr.blocks' (list)."""
    if not isinstance(payload, dict):
        return False

    if "meta" not in payload or "ocr" not in payload:
        return False

    ocr = payload.get("ocr")
    if not isinstance(ocr, dict):
        return False

    blocks = ocr.get("blocks")
    if not isinstance(blocks, list):
        return False

    return True








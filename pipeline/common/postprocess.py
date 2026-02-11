#!/usr/bin/env python3
"""Postprocessing utilities: text normalization and header/footer classification."""

import logging
import re
from typing import Dict, Any, List, Optional


def normalize_text(text: str) -> str:
    """Normalize text: strip whitespace, normalize newlines, collapse multiple spaces per line."""
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


def classify_header_footer_heuristic(
    blocks: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None
) -> List[Dict[str, Any]]:
    """Classify blocks as page-header or page-footer based on position, size, and content patterns."""
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




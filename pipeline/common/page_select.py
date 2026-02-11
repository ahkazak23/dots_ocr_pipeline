#!/usr/bin/env python3
"""Page selection parsing utilities."""

from typing import List, Optional


def parse_page_selection(
    pages: Optional[List[int]],
    page_range: Optional[str],
    total_pages: int
) -> List[int]:
    """Parse page selection arguments (1-based input) and return 0-based page indices."""
    def normalize(idx: int) -> int:
        """Convert 1-based page number to 0-based index."""
        return idx - 1

    if pages:
        # Filter pages to valid range [1, total_pages] and convert to 0-based
        return [normalize(p) for p in pages if 1 <= p <= total_pages]

    if page_range:
        try:
            start_s, end_s = page_range.split("-")
            start, end = int(start_s), int(end_s)
            # Clamp to valid range
            start = max(start, 1)
            end = min(end, total_pages)
            if start <= end:
                # Return 0-based indices for the inclusive range
                return list(range(normalize(start), normalize(end) + 1))
        except ValueError:
            pass

    # Default: all pages
    return list(range(total_pages))



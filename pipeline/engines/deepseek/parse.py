#!/usr/bin/env python3
"""DeepSeek OCR grounded output parsing."""

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from lsextractor.pipeline.common.postprocess import normalize_text


REF_RE = re.compile(
    r"<\|ref\|>(?P<label>.*?)<\|/ref\|>.*?<\|det\|>(?P<det>\[.*?\])<\|/det\|>\s*",
    re.DOTALL,
)

LABEL_TO_TYPE = {
    "text": "text_block",
    "title": "title",
    "table": "table",
    "table_caption": "table_caption",
    "image": "image",
    "diagram": "diagram",
    "header": "page_header",
    "footer": "page_footer",
    "page-header": "page_header",
    "page-footer": "page_footer",
}


def _parse_det(det_text: str):
    """Parse detection coordinates from string."""
    det_text = det_text.strip()
    try:
        return ast.literal_eval(det_text)
    except Exception:
        try:
            return json.loads(det_text)
        except Exception:
            return None


def _det_to_bbox_normalized(det_coords, img_w: int, img_h: int) -> List[float]:
    """Convert detection coordinates to normalized bbox [x1, y1, x2, y2] in 0-1 range."""
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

    norm_x1 = px1 / img_w if img_w > 0 else 0.0
    norm_y1 = py1 / img_h if img_h > 0 else 0.0
    norm_x2 = px2 / img_w if img_w > 0 else 0.0
    norm_y2 = py2 / img_h if img_h > 0 else 0.0

    return [
        max(0.0, min(1.0, norm_x1)),
        max(0.0, min(1.0, norm_y1)),
        max(0.0, min(1.0, norm_x2)),
        max(0.0, min(1.0, norm_y2)),
    ]


def _split_caption_and_table(content: str) -> Tuple[str, str, str]:
    """Split table_caption content into (caption_text, table_markup, table_type)."""
    if not content:
        return "", "", ""

    if "<table" in content.lower():
        parts = content.split("<table", 1)
        caption_text = parts[0].strip()
        table_html = "<table" + parts[1] if len(parts) > 1 else ""
        return caption_text, table_html, "html"

    lines = content.split("\n")
    table_start_idx = -1

    for i, line in enumerate(lines):
        if "|" in line:
            if i + 1 < len(lines) and "-" in lines[i + 1] and "|" in lines[i + 1]:
                table_start_idx = i
                break

    if table_start_idx >= 0:
        caption_lines = lines[:table_start_idx]
        table_lines = lines[table_start_idx:]

        table_end_idx = len(table_lines)
        for i, line in enumerate(table_lines):
            if line.strip() and "|" not in line:
                table_end_idx = i
                break

        caption_text = "\n".join(caption_lines).strip()
        table_markdown = "\n".join(table_lines[:table_end_idx]).strip()
        return caption_text, table_markdown, "markdown"

    return content.strip(), "", ""


def try_parse_markdown_table(md: str) -> Optional[Dict[str, Any]]:
    """Parse markdown table into structured data, returning {"columns": [...], "rows": [[...], ...]} or None."""
    if not md or "|" not in md:
        return None

    lines = [line.strip() for line in md.split("\n") if line.strip()]
    if len(lines) < 2:
        return None

    separator_idx = -1
    for i, line in enumerate(lines):
        if "|" in line and "-" in line:
            cleaned = line.replace("|", "").replace("-", "").replace(":", "").strip()
            if not cleaned or all(c in "- |:" for c in line):
                separator_idx = i
                break

    if separator_idx < 1:
        return None

    header_line = lines[separator_idx - 1]
    columns = [cell.strip() for cell in header_line.split("|")]
    if columns and not columns[0]:
        columns = columns[1:]
    if columns and not columns[-1]:
        columns = columns[:-1]

    if not columns:
        return None

    col_count = len(columns)

    rows = []
    for i in range(separator_idx + 1, len(lines)):
        row_line = lines[i]
        if "|" not in row_line:
            continue
        cells = [cell.strip() for cell in row_line.split("|")]
        if cells and not cells[0]:
            cells = cells[1:]
        if cells and not cells[-1]:
            cells = cells[:-1]

        while len(cells) < col_count:
            cells.append("")
        cells = cells[:col_count]

        rows.append(cells)

    return {"columns": columns, "rows": rows}


def try_parse_html_table(html: str) -> Optional[Dict[str, Any]]:
    """Parse HTML table into structured data, returning {"headers": [...], "rows": [[...], ...]} or None."""
    if not html or "<table" not in html.lower():
        return None

    tr_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.IGNORECASE | re.DOTALL)
    tr_matches = tr_pattern.findall(html)

    if not tr_matches:
        return None

    headers = []
    rows = []

    for i, tr_content in enumerate(tr_matches):
        th_pattern = re.compile(r'<th[^>]*>(.*?)</th>', re.IGNORECASE | re.DOTALL)
        td_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.IGNORECASE | re.DOTALL)

        th_matches = th_pattern.findall(tr_content)
        td_matches = td_pattern.findall(tr_content)

        if th_matches and i == 0:
            headers = [_clean_html(cell) for cell in th_matches]
        elif td_matches:
            row = [_clean_html(cell) for cell in td_matches]
            rows.append(row)
        elif th_matches and not headers:
            headers = [_clean_html(cell) for cell in th_matches]

    if headers or rows:
        return {"headers": headers, "rows": rows}

    return None


def _clean_html(text: str) -> str:
    """Remove HTML tags and clean up text."""
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&amp;', '&')
    text = text.replace('&quot;', '"')
    text = ' '.join(text.split())
    return text.strip()


def _has_grounding_tags(text: str) -> bool:
    """Check if text contains DeepSeek grounding tags."""
    return bool(text and "<|ref|>" in text and "<|det|>" in text)


def _clean_model_debug_output(text: str) -> str:
    """Remove model debug statistics from captured output."""
    if not text:
        return text

    lines = text.split('\n')

    last_separator_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if '=' * 20 in lines[i]:
            last_separator_idx = i
            break

    if last_separator_idx > 0:
        for i in range(last_separator_idx - 1, -1, -1):
            if '=' * 20 in lines[i]:
                cleaned_lines = lines[:i]
                return '\n'.join(cleaned_lines).strip()

    return text


def parse_grounded_stdout(
    captured: str,
    image_path: Path,
    logger: logging.Logger,
    page_id: str = "",
    store_raw: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any], List[int]]:
    """Parse grounded model output, returning (page_ocr_dict, parse_diag, resolution)."""
    img = Image.open(image_path)
    if image_path.suffix.lower() == ".png" and image_path.name.startswith("page_") and img.mode != "RGB":
        img = img.convert("RGB")

    img_w, img_h = img.size
    resolution = [img_w, img_h]

    logger.debug(
        "[%s] Parsing grounded output: image=%s (%dx%d), captured_len=%d",
        page_id or "UNKNOWN",
        image_path.name,
        img_w,
        img_h,
        len(captured),
    )

    matches = list(REF_RE.finditer(captured))
    blocks: List[Dict[str, Any]] = []
    bbox_parse_failures = 0
    malformed_tags = 0

    if "<|ref|>" in captured:
        ref_count = captured.count("<|ref|>")
        det_count = captured.count("<|det|>")
        det_close_count = captured.count("<|/det|>")

        if ref_count != det_count or det_count != det_close_count:
            logger.warning(
                "[MALFORMED] Unbalanced grounding tags on %s: ref=%d, det=%d, /det=%d",
                image_path.name, ref_count, det_count, det_close_count
            )
            malformed_tags = abs(ref_count - det_count) + abs(det_count - det_close_count)

        if "eee" in captured.lower():
            eee_count = captured.lower().count("eee")
            if eee_count > 3:
                logger.warning(
                    "[MALFORMED] Detected 'eee' pattern (%d occurrences) - likely redacted signatures/watermarks",
                    eee_count
                )

    for i, match in enumerate(matches):
        label = match.group("label").strip()
        det_text = match.group("det").strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(captured)
        content = captured[start:end].strip()

        original_content = content
        content_had_leading_garbage = False

        while content:
            ref_close_match = re.match(r'^([^<]*)<\|/ref\|>', content)
            if ref_close_match:
                content = content[ref_close_match.end():].strip()
                content_had_leading_garbage = True
                continue

            if content.startswith("<|ref|>"):
                content = content[len("<|ref|>"):].strip()
                content_had_leading_garbage = True
                continue

            if content.startswith("<|/ref|>"):
                content = content[len("<|/ref|>"):].strip()
                content_had_leading_garbage = True
                continue

            break

        if content_had_leading_garbage:
            logger.warning(
                "[MALFORMED] Block %d: Content started with garbage+ref tags. "
                "Removed prefix '%s...' → remaining content_len=%d",
                len(blocks), original_content[:50], len(content)
            )
            malformed_tags += 1
        elif ("<|/ref|>" in content or "<|ref|>" in content) and content:
            logger.warning(
                "[MALFORMED] Block %d: Content contains embedded ref tags: '%s...'",
                len(blocks), content[:100]
            )
            malformed_tags += 1

        is_corrupted = (
                not content or
                content.startswith("]]<|/det|>") or
                "eee eee" in content.lower() or
                len(content) < 2
        )

        if is_corrupted:
            logger.warning(
                "[MALFORMED] Block %d has corrupted content: label=%s, content_preview=%s",
                len(blocks), label, content[:100]
            )

        try:
            det_coords = _parse_det(det_text)
            bbox = _det_to_bbox_normalized(det_coords, img_w, img_h)

            if bbox == [0.0, 0.0, 0.0, 0.0] and det_text:
                logger.warning(
                    "[MALFORMED] Zero bbox from det_text=%s | label=%s | content_preview=%s",
                    det_text[:100], label, content[:50]
                )
        except Exception as exc:
            bbox = [0.0, 0.0, 0.0, 0.0]
            bbox_parse_failures += 1
            logger.warning(
                "bbox parse failed for block %d on %s: %s: %s | det_text=%s",
                len(blocks),
                image_path.name,
                type(exc).__name__,
                exc,
                det_text[:100],
            )

        block_type = LABEL_TO_TYPE.get(label, "text_block")

        if block_type == "table":
            table_data = try_parse_html_table(content)
            if table_data is None:
                table_data = try_parse_markdown_table(content)
            parsed_payload = {"data": table_data, "text": ""}
        elif block_type == "table_caption":
            caption_text, table_markup, table_type = _split_caption_and_table(content)

            table_data = None
            if table_type == "html":
                table_data = try_parse_html_table(table_markup)
            elif table_type == "markdown":
                table_data = try_parse_markdown_table(table_markup)

            parsed_payload = {
                "data": table_data,
                "text": normalize_text(caption_text),
                "_has_table_html": bool(table_markup),
                "_table_html": table_markup
            }
        elif block_type == "image":
            parsed_payload = {"data": None, "text": ""}
        else:
            parsed_payload = {"data": None, "text": normalize_text(content)}

        legacy_block = {
            "type": block_type,
            "bbox": bbox,
            "extraction_origin": "deepseek-ocr",
            "extraction_response": content,
            "extraction_response_parsed": parsed_payload,
            "id": f"blk_{len(blocks) + 1}",
            "order": len(blocks),
        }

        if store_raw:
            try:
                det_coords_parsed = _parse_det(det_text) if det_text else None
            except:
                det_coords_parsed = None

            legacy_block["raw_label"] = label
            legacy_block["raw_det"] = det_text
            legacy_block["raw_det_parsed"] = det_coords_parsed
            legacy_block["raw_bbox_parse_failed"] = bbox == [0.0, 0.0, 0.0, 0.0] and bool(det_text)
            legacy_block["raw_match_index"] = i
            legacy_block["raw_is_corrupted"] = is_corrupted

        blocks.append(legacy_block)

    # Post-process: separate table and caption payloads
    processed_blocks = []
    i = 0
    while i < len(blocks):
        block = blocks[i]

        if block["type"] == "table" and i + 1 < len(blocks) and blocks[i + 1]["type"] == "table_caption":
            table_block = block
            caption_block = blocks[i + 1]

            table_parsed = table_block.get("extraction_response_parsed", {})
            table_has_data = table_parsed.get("data") is not None

            caption_parsed = caption_block.get("extraction_response_parsed", {})
            caption_text = caption_parsed.get("text", "")
            caption_has_table_data = caption_parsed.get("data") is not None
            table_html_from_caption = caption_parsed.get("_table_html", "")

            if caption_has_table_data and not table_has_data:
                table_block["extraction_response_parsed"]["data"] = caption_parsed.get("data")
                if table_html_from_caption:
                    table_block["extraction_response"] = table_html_from_caption
                logger.debug(
                    "[%s] Moved table data from caption to table block (table was empty)",
                    page_id or "UNKNOWN"
                )

            caption_block["extraction_response_parsed"] = {"data": None, "text": caption_text}
            caption_block["extraction_response"] = caption_text

            if "_has_table_html" in caption_parsed:
                del caption_parsed["_has_table_html"]
            if "_table_html" in caption_parsed:
                del caption_parsed["_table_html"]

            processed_blocks.append(table_block)
            processed_blocks.append(caption_block)
            i += 2
            logger.debug(
                "[%s] Separated table + caption: caption_text='%s', table_has_data=%s",
                page_id or "UNKNOWN", caption_text[:50] if caption_text else "",
                caption_has_table_data or table_has_data
            )

        elif block["type"] == "table_caption" and i + 1 < len(blocks) and blocks[i + 1]["type"] == "table":
            caption_block = block
            table_block = blocks[i + 1]

            caption_parsed = caption_block.get("extraction_response_parsed", {})
            caption_text = caption_parsed.get("text", "")
            caption_has_table_data = caption_parsed.get("data") is not None
            table_html_from_caption = caption_parsed.get("_table_html", "")

            table_parsed = table_block.get("extraction_response_parsed", {})
            table_has_data = table_parsed.get("data") is not None

            if caption_has_table_data and not table_has_data:
                table_block["extraction_response_parsed"]["data"] = caption_parsed.get("data")
                if table_html_from_caption:
                    table_block["extraction_response"] = table_html_from_caption
                logger.debug(
                    "[%s] Moved table data from caption to table (reverse order)",
                    page_id or "UNKNOWN"
                )

            caption_block["extraction_response_parsed"] = {"data": None, "text": caption_text}
            caption_block["extraction_response"] = caption_text

            if "_has_table_html" in caption_parsed:
                del caption_parsed["_has_table_html"]
            if "_table_html" in caption_parsed:
                del caption_parsed["_table_html"]

            processed_blocks.append(caption_block)
            processed_blocks.append(table_block)
            i += 2
            logger.debug(
                "[%s] Separated caption + table (reverse): caption='%s'",
                page_id or "UNKNOWN", caption_text[:50] if caption_text else ""
            )

        elif block["type"] == "table_caption":
            caption_parsed = block.get("extraction_response_parsed", {})
            caption_has_table_data = caption_parsed.get("data") is not None

            if caption_has_table_data:
                block["type"] = "table"
                table_html = caption_parsed.get("_table_html", "")
                if table_html:
                    block["extraction_response"] = table_html
                if "_has_table_html" in block["extraction_response_parsed"]:
                    del block["extraction_response_parsed"]["_has_table_html"]
                if "_table_html" in block["extraction_response_parsed"]:
                    del block["extraction_response_parsed"]["_table_html"]
                processed_blocks.append(block)
                logger.debug(
                    "[%s] Converted standalone table_caption to table: caption=%s",
                    page_id or "UNKNOWN",
                    caption_parsed.get("text", "")[:50]
                )
            else:
                logger.warning(
                    "[%s] Found standalone table_caption without table data: %s",
                    page_id or "UNKNOWN",
                    caption_parsed.get("text", "")[:50]
                )
                processed_blocks.append(block)
            i += 1

        else:
            processed_blocks.append(block)
            i += 1

    blocks = processed_blocks

    parse_diag = {
        "img_w": img_w,
        "img_h": img_h,
        "match_count": len(matches),
        "block_count": len(blocks),
        "bbox_parse_failures": bbox_parse_failures,
        "malformed_tags": malformed_tags,
    }

    return {"ocr": {"blocks": blocks}}, parse_diag, resolution


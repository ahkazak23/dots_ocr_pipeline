#!/usr/bin/env python3
"""
OCR Pipeline for Amazon Textract (AnalyzeDocument)

Accepts images and PDFs. For PDFs, renders each page to an image, then runs Textract OCR
and returns block-based layout JSON per page.

Install:
  pip install boto3 pillow pymupdf

Env (one of the common AWS credential setups):
  export AWS_REGION="us-east-1"
  export AWS_ACCESS_KEY_ID="..."
  export AWS_SECRET_ACCESS_KEY="..."
  # or use ~/.aws/credentials, SSO, IAM role, etc.
"""

from __future__ import annotations

import json
import os
import re
import sys
import fitz  # PyMuPDF
import boto3
import time
import lsextractor.evaluate.cost.cost_evaluation as ce
import lsextractor.io.file as fi
import lsextractor.utils.model_config as CFG
import lsextractor.io.json_out as wrt


from lsextractor.utils.bbox import ltwh_to_xyxy, choose_and_normalize, check_normalized
from lsextractor.utils.drawing import draw_page_bbox_save
from datetime import datetime
from lsextractor.io.registry import register_prompt
from PIL import Image
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from timeit import default_timer as timer


# -----------------------------
# Response schema (supports image=1 page, pdf=multi-page)
# -----------------------------
PAGE_OCR_SCHEMA = {
            "type": "OBJECT",
            "properties": {
                "blocks": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "type": {
                                "type": "STRING",
                                "enum": ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
                            },                            
                            "bbox": {
                                "type": "ARRAY",
                                "items": {"type": "NUMBER"},
                                "minItems": 4,
                                "maxItems": 4
                            },
                            "extraction_origin" : {"type": "STRING"},
                            "extraction_response": {"type": "STRING"},
                            "extraction_response_parsed": {
                                "type": "OBJECT",
                                "properties": {
                                    # IMPORTANT:
                                    # - If you want arbitrary JSON here, the safest route is STRING(nullable)
                                    #   and store JSON-serialized text.
                                    "data": {"type": "STRING", "nullable": True},
                                    "text": {"type": "STRING"}
                                },
                                "required": ["data", "text"]
                            }
                        },
                        "required": [
                            "type", "bbox", "extraction_origin", 
                            "extraction_response", "extraction_response_parsed"
                        ]
                    }
                }
            },
            "required": ["blocks"]
        }


SYSTEM_INSTRUCTIONS = """
You are an OCR + document layout understanding engine.

You will receive an image of a single page.

Return ONLY valid JSON matching the provided schema for page_ocr (ocr.blocks).

Block rules:
- Detect logical layout blocks on the page and output them in strict reading order.
- Layout categories: ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
- Bbox format: [x1, y1, x2, y2] raw pixel values.
- extraction_response: raw OCR output in markdown when applicable.
- extraction_response_parsed:
    - text: readable text or caption
    - data:
        - null for paragraphs/headers
        - table: JSON (e.g., {"header":[...], "rows":[[...]...]})
        - chart: JSON (e.g., {"chart_type":"...", "x_axis":"...", "y_axis":"...", "series":[...]}) if readable
        - image: JSON metadata if inferable, otherwise null

extraction_response_parsed.data:
- Use null when there is no structured data.
- Otherwise store structured data as a JSON-serialized STRING (not an object).
  Example for table: {"header":[...],"rows":[...]} serialized into a string.
        

Do NOT invent/hallucinate content. If unreadable, use empty strings/arrays or null.
Return JSON only (no markdown fences).
"""


_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

TIME_COUNTER_PRECISION = 4


def coerce_json(text: str) -> Dict[str, Any]:
    cleaned = _JSON_FENCE_RE.sub("", text).strip()
    return json.loads(cleaned)



def build_client(cfg: CFG.TextractConfig):    

    session = boto3.Session(aws_access_key_id=cfg.access_key,
                            aws_secret_access_key=cfg.secret_key,
                            region_name=cfg.region)

    return session.client(cfg.model)


def is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def guess_input_type(path: Path) -> str:
    return "pdf" if is_pdf(path) else "image"


def load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def render_pdf_to_pil_images(pdf_path: Path, dpi: int) -> List[Image.Image]:
    """
    Render each page to a PIL Image using PyMuPDF.
    dpi controls raster quality. 200-300 is typical for OCR.
    """
    doc = fitz.open(pdf_path)
    images: List[Image.Image] = []

    zoom = dpi / 72.0  # PDF points are 72 dpi
    mat = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        mode = "RGB" if pix.n < 4 else "RGBA"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)  # pyright: ignore[reportArgumentType]
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)

    doc.close()
    return images


# -----------------------------
# Textract helpers
# -----------------------------
def _pil_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    import io
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _bbox_pixels(bb_norm: Dict[str, float], page_w: int, page_h: int) -> List[float]:
    # Textract BoundingBox is normalized [0..1]: Left, Top, Width, Height
    left = float(bb_norm.get("Left", 0.0)) * page_w
    top = float(bb_norm.get("Top", 0.0)) * page_h
    width = float(bb_norm.get("Width", 0.0)) * page_w
    height = float(bb_norm.get("Height", 0.0)) * page_h
    return [left, top, width, height]


def _build_id_map(blocks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {b["Id"]: b for b in blocks if "Id" in b}


def _children_ids(block: Dict[str, Any], rel_type: str) -> List[str]:
    rels = block.get("Relationships", []) or []
    out: List[str] = []
    for r in rels:
        if r.get("Type") == rel_type:
            out.extend(r.get("Ids", []) or [])
    return out


def _get_words_text(block: Dict[str, Any], id_map: Dict[str, Dict[str, Any]]) -> str:
    words: List[str] = []
    for cid in _children_ids(block, "CHILD"):
        c = id_map.get(cid)
        if not c:
            continue
        bt = c.get("BlockType")
        if bt == "WORD":
            t = c.get("Text", "")
            if t:
                words.append(t)
        elif bt == "SELECTION_ELEMENT":
            if c.get("SelectionStatus") == "SELECTED":
                words.append("[x]")
            else:
                words.append("[ ]")
    return " ".join(words).strip()


def _looks_like_header(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    # Simple heuristic: short uppercase-ish lines or ending with colon
    if len(t) <= 60 and (t.isupper() or t.endswith(":")):
        return True
    return False


def _union_bbox(b1: List[float], b2: List[float]) -> List[float]:
    l1, t1, w1, h1 = b1
    l2, t2, w2, h2 = b2
    r1, btm1 = l1 + w1, t1 + h1
    r2, btm2 = l2 + w2, t2 + h2
    left = min(l1, l2)
    top = min(t1, t2)
    right = max(r1, r2)
    bottom = max(btm1, btm2)
    return [left, top, right - left, bottom - top]


def _group_lines_into_paragraphs(
    line_blocks: List[Dict[str, Any]],
    id_map: Dict[str, Dict[str, Any]],
    page_w: int,
    page_h: int,
    max_vgap_px: float = 18.0,
) -> List[Dict[str, Any]]:
    """
    Groups LINE blocks into paragraphs using vertical gap in pixels.
    """
    enriched: List[Tuple[str, List[float], str]] = []
    for ln in line_blocks:
        bb = (ln.get("Geometry", {}) or {}).get("BoundingBox", {}) or {}
        bbox_px = _bbox_pixels(bb, page_w, page_h)
        text = ln.get("Text") or _get_words_text(ln, id_map)
        enriched.append((ln["Id"], bbox_px, text))

    # Sort: top-to-bottom then left-to-right
    enriched.sort(key=lambda x: (x[1][1], x[1][0]))

    paras: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    prev_bbox: Optional[List[float]] = None

    for lid, bbox, text in enriched:
        if cur is None:
            cur = {"bbox": bbox, "texts": [text], "lines": [lid]}
            prev_bbox = bbox
            continue

        assert prev_bbox is not None
        prev_bottom = prev_bbox[1] + prev_bbox[3]
        vgap = bbox[1] - prev_bottom

        if vgap > max_vgap_px:
            paras.append(
                {
                    "bbox": cur["bbox"],
                    "text": "\n".join([t for t in cur["texts"] if t]).strip(),
                    "lines": cur["lines"],
                }
            )
            cur = {"bbox": bbox, "texts": [text], "lines": [lid]}
            prev_bbox = bbox
        else:
            cur["bbox"] = _union_bbox(cur["bbox"], bbox)
            cur["texts"].append(text)
            cur["lines"].append(lid)
            prev_bbox = bbox

    if cur is not None:
        paras.append(
            {
                "bbox": cur["bbox"],
                "text": "\n".join([t for t in cur["texts"] if t]).strip(),
                "lines": cur["lines"],
            }
        )

    return [p for p in paras if p.get("text")]


def _extract_tables(
    blocks: List[Dict[str, Any]],
    id_map: Dict[str, Dict[str, Any]],
    page_w: int,
    page_h: int,
) -> List[Dict[str, Any]]:
    """
    Extract table grid from Textract TABLE/CELL blocks.
    Returns list of:
      { "bbox": [x,y,w,h], "table": {"header":[...], "rows":[[...]...]} }
    """
    tables_out: List[Dict[str, Any]] = []

    table_blocks = [b for b in blocks if b.get("BlockType") == "TABLE"]
    for tb in table_blocks:
        tb_bb = (tb.get("Geometry", {}) or {}).get("BoundingBox", {}) or {}
        tb_bbox_px = _bbox_pixels(tb_bb, page_w, page_h)        

        cell_ids = _children_ids(tb, "CHILD")
        cells: List[Dict[str, Any]] = []
        max_r, max_c = 0, 0

        for cid in cell_ids:
            cb = id_map.get(cid)
            if not cb or cb.get("BlockType") != "CELL":
                continue

            r = int(cb.get("RowIndex", 0))
            c = int(cb.get("ColumnIndex", 0))
            rs = int(cb.get("RowSpan", 1))
            cs = int(cb.get("ColumnSpan", 1))
            max_r = max(max_r, r + rs - 1)
            max_c = max(max_c, c + cs - 1)

            text = _get_words_text(cb, id_map)
            cells.append({"row": r, "col": c, "rowSpan": rs, "colSpan": cs, "text": text})

        grid = [["" for _ in range(max_c)] for _ in range(max_r)]
        for cell in cells:
            r0 = max(cell["row"] - 1, 0)
            c0 = max(cell["col"] - 1, 0)
            if 0 <= r0 < max_r and 0 <= c0 < max_c:
                grid[r0][c0] = cell["text"]

        # Simple "header + rows" view
        header = grid[0] if grid else []
        rows = grid[1:] if len(grid) > 1 else []

        tables_out.append(
            {
                "bbox": tb_bbox_px,
                "table": {"header": header, "rows": rows},
                "grid": grid,
            }
        )

    return tables_out


def _grid_to_markdown(grid: List[List[str]]) -> str:
    if not grid:
        return ""
    width = max((len(r) for r in grid), default=0)
    rows = [r + [""] * (width - len(r)) for r in grid]

    header = rows[0]
    sep = ["---"] * width
    md: List[str] = []
    md.append("| " + " | ".join((c or "").replace("\n", " ").strip() for c in header) + " |")
    md.append("| " + " | ".join(sep) + " |")
    for r in rows[1:]:
        md.append("| " + " | ".join((c or "").replace("\n", " ").strip() for c in r) + " |")
    return "\n".join(md)


def extract_page_blocks(
    client,
    pipe_config: CFG.TestConfig,
    page_image: Image.Image,
    registry_cls: ce.CostRegistry,
    doc_info: tuple
) -> Dict[str, Any]:

    start = timer()

    page_w, page_h = page_image.size
    img_bytes = _pil_to_jpeg_bytes(page_image)

    resp = client.analyze_document(
        Document={"Bytes": img_bytes},
        FeatureTypes=pipe_config.model_cfg.features, # type: ignore
    )

    process_time = round(timer() - start, TIME_COUNTER_PRECISION)
    ce.register_metrics(cls=registry_cls, 
                        process_time=process_time,                         
                        doc_info=doc_info,
                        token_metadata=None)
    ce.save_to_file(registry_cls, pipe_config.eval_out_dir)

    
    blocks = resp.get("Blocks", []) or []
    id_map = _build_id_map(blocks)

    # Build our schema output:
    out_blocks: List[Dict[str, Any]] = []

    # Paragraphs/headers from LINE blocks
    line_blocks = [b for b in blocks if b.get("BlockType") == "LINE"]
    paras = _group_lines_into_paragraphs(line_blocks, id_map, page_w, page_h)

    for p in paras:
        text = (p.get("text") or "").strip()
        if not text:
            continue
        btype = "header" if _looks_like_header(text) else "paragraph"
        out_blocks.append(
            {
                "type": btype,
                "bbox": p["bbox"],
                "extraction_origin": registry_cls.model,
                "extraction_response": text,
                "extraction_response_parsed": {"data": None, "text": text},
            }
        )

    # Tables
    tables = _extract_tables(blocks, id_map, page_w, page_h)
    for t in tables:
        md = _grid_to_markdown(t["grid"])
        # data must be JSON-serialized STRING (per your schema comment)
        data_str = json.dumps(t["table"], ensure_ascii=False)
        out_blocks.append(
            {
                "type": "table",
                "bbox": t["bbox"],
                "extraction_origin": registry_cls.model,
                "extraction_response": md,
                "extraction_response_parsed": {"data": data_str, "text": ""},
            }
        )

    
    return {"blocks": out_blocks}

def call_with_fallback(client, 
                        pipe_config: CFG.TestConfig, 
                        page_image: Image.Image,                                            
                        doc_info: tuple,
                        registry_cls: ce.CostRegistry,
                        result: Dict) -> Dict:
    
    for attempt in range(pipe_config.max_retries):
        try:
            page_ocr = extract_page_blocks(
                client=client,
                pipe_config=pipe_config,
                page_image=page_image,
                registry_cls=registry_cls,
                doc_info=doc_info
            )
            for bi, block in enumerate(page_ocr['blocks']):
                block['bbox'] = ltwh_to_xyxy(block['bbox'])
                if not check_normalized(block['bbox']):
                    page_ocr['blocks'][bi]['bbox'] = choose_and_normalize(block['bbox'], 
                                                                    page_image.size[0], 
                                                                    page_image.size[1])            
            result["ocr"] = page_ocr
            return result
        except Exception as e:
            msg = str(e)
            result["meta"]["warnings"].append(f"Page {doc_info[0]} failed: {type(e).__name__}: {e}")
            result["ocr"] = {"blocks": []}            
            if len(registry_cls.time)>0 and registry_cls.time[-1] > 90:
                print(f'Gemini request failed, skipping {doc_info[1]}')
            if "503" in msg or "UNAVAILABLE" in msg or "429" in msg:
                print(f'Error occured during API call, retry attempt: {attempt}')                
                time.sleep(3)
                continue
    return result

def process_input(
    client,
    pipe_config: CFG.TestConfig,
    input_path: Path,
    registry_cls: ce.CostRegistry
):

    input_type = guess_input_type(input_path)
    pages: List[Image.Image]

    if input_type == "pdf":
        pages = render_pdf_to_pil_images(input_path, dpi=pipe_config.dpi)
    else:
        pages = [load_image(input_path)]

    current_document = fi.get_file_name(input_path)

    for page_index, page_image in enumerate(pages):
        print(f'Page {page_index+1}')
        result: Dict[str, Any] = {
            "meta": {
                "document_id": input_path.name,
                "page_id": page_index,
                "page_resolution": list(page_image.size),
                "warnings": [],
            },
            "ocr": {},
        }
        
        doc_info = (page_index, input_path.name)
        if wrt.skip_if_exists(pipe_config=pipe_config,
                                doc_info=(page_index, current_document)):
            print(f'doc/page exists, skipping...')
            continue

        result = call_with_fallback(client=client,
                            pipe_config=pipe_config,
                            page_image=page_image,
                            doc_info=doc_info,
                            registry_cls=registry_cls,
                            result=result)
        
        wrt.write_page_json(pipe_config=pipe_config,
                            doc_info=doc_info,
                            result=result)
        
        draw_page_bbox_save(img=page_image,
                            blocks=result['ocr']['blocks'],
                            color=(0, 0, 255, 100),
                            out_path=Path(pipe_config.out_dir),
                            doc_info=doc_info)


def main() -> int:
    all_files = fi.get_all_files("./output/images/gleevec")    

    cfg = CFG.TextractConfig(model='textract',
                        region=os.environ.get("AWS_DEFAULT_REGION"),
                        access_key=os.environ.get("AWS_ACCESS_KEY"),
                        secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                        features=['TABLES', 'FORMS'])

    pipe_config = CFG.TestConfig(model=cfg.model,
                             model_cfg=cfg,
                             input_files=all_files)

    TextractCostInfo = ce.PageCostInfo(
        page_price=70.0,
        pricing_unit=1000.0
    )

    TextractRegistry = ce.CostRegistry(
        model=pipe_config.model,
        info=TextractCostInfo,
        registry=ce.RegistryInfo(model=pipe_config.model)
    )

    register_prompt(SYSTEM_INSTRUCTIONS, cfg.model, str(datetime.now()))
    client = build_client(cfg)
    print("boto3 Textract client built successfully")

    for file_index, inp in enumerate(pipe_config.input_files):
        print(f"\n**DOCUMENT = {fi.get_file_name(inp)}, {file_index}/{len(pipe_config.input_files)}**")
        input_path = Path(inp).expanduser().resolve()
        if not input_path.exists():
            print(f"ERROR: Not found: {input_path}", file=sys.stderr)
            continue

        process_input(
            client=client,
            pipe_config=pipe_config,
            input_path=input_path,
            registry_cls=TextractRegistry
        )

    return 0

if __name__ == "__main__":
    main()

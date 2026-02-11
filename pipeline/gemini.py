#!/usr/bin/env python3
"""
OCR Pipeline for Gemini-2.5-Flash-Preview-09-2025

Accepts images and PDFs. For PDFs, renders each page to an image, then runs Gemini OCR
and returns block-based layout JSON per page.

Install:
  pip install google-genai pillow pymupdf

Env:
  export GEMINI_API_KEY="YOUR_KEY"   (or GOOGLE_API_KEY)
"""

from __future__ import annotations

import json
import os
import re
import sys
import fitz  # PyMuPDF
import time
import google.genai as genai
import lsextractor.io.json_out as wrt
import lsextractor.evaluate.cost.cost_evaluation as ce
import lsextractor.io.file as fi
import lsextractor.utils.model_config as CFG
from lsextractor.utils.bbox import check_normalized, normalize_xyxy_1000

from datetime import datetime
from lsextractor.io.registry import register_prompt
from PIL import Image
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
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

def build_client(cfg: CFG.GeminiConfig) -> genai.Client:    
    return genai.Client(api_key=cfg.api_key)


def is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def guess_input_type(path: Path) -> str:
    return "pdf" if is_pdf(path) else "image"


def load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    # Force load to catch errors early
    img.load()
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
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples) # pyright: ignore[reportArgumentType]
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)

    doc.close()
    return images


def extract_page_blocks(
    client: genai.Client,
    pipe_config: CFG.TestConfig,
    page_image: Image.Image,
    registry_cls: ce.CostRegistry,
    doc_info: tuple
) -> tuple[Dict[str, Any], str]:
    user_text = (
        "Perform OCR + layout block extraction for this image. "
        "Return JSON only matching the schema."
    )
    start = timer()
    response = client.models.generate_content(
        model=pipe_config.model_cfg.model,
        contents=[user_text, page_image],
        config={
            "system_instruction": SYSTEM_INSTRUCTIONS,
            "temperature": pipe_config.temperature,
            "max_output_tokens": pipe_config.model_cfg.max_output_tokens, # pyright: ignore[reportAttributeAccessIssue]
            "response_mime_type": "application/json",
            "response_schema": PAGE_OCR_SCHEMA,
        },
    )
    
    process_time = round(timer()-start, TIME_COUNTER_PRECISION)
    ce.register_metrics(cls=registry_cls, 
                        process_time=process_time,                         
                        doc_info=doc_info,
                        token_metadata=response.usage_metadata)
    ce.save_to_file(registry_cls, pipe_config.eval_out_dir)
    raw_text = getattr(response, "text", None)
    if not raw_text:
        raise RuntimeError("No response.text returned by the SDK; inspect the response object.")
    return (coerce_json(raw_text), response.model_version) # pyright: ignore[reportReturnType]


def call_with_fallback(client: genai.Client, 
                        pipe_config: CFG.TestConfig, 
                        page_image: Image.Image,                                            
                        doc_info: tuple,
                        registry_cls: ce.CostRegistry,
                        result: Dict) -> Dict:
    
    for attempt in range(pipe_config.max_retries):
        try:
            page_ocr, model_version = extract_page_blocks(client=client, 
                                                        pipe_config=pipe_config, 
                                                        page_image=page_image,                                            
                                                        doc_info=doc_info,
                                                        registry_cls=registry_cls)
            for bi, block in enumerate(page_ocr['blocks']):
                block['extraction_origin'] = model_version
                page_ocr['blocks'][bi] = block
                if not check_normalized(block['bbox']):
                    page_ocr['blocks'][bi] = normalize_xyxy_1000(block['bbox'], 
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
    client: genai.Client,
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

    for page_index, page_image in enumerate(pages):
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
                                doc_info=(page_index, input_path.name)):
            print(f'doc/page exists, skipping...')
            continue        
        call_with_fallback(client=client, 
                            pipe_config=pipe_config, 
                            page_image=page_image,                                            
                            doc_info=doc_info,
                            registry_cls=registry_cls,
                            result=result)
        
        wrt.write_page_json(pipe_config=pipe_config,
                            doc_info=doc_info,
                            result=result)        

    


def main() -> int:

    api_key = os.environ.get("GEMINI_API_KEY")    
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.", file=sys.stderr)
        return 2

        
    all_files = fi.get_all_files('./output/omnidocbench/images')   

    cfg = CFG.GeminiConfig(
        model='gemini-2.5-flash-preview-09-2025',
        api_key=api_key,        
        temperature=0.1,
        max_output_tokens=16384
    )
    
    pipe_config = CFG.TestConfig(model=cfg.model,
                             model_cfg=cfg,
                             input_files=all_files)

    GeminiCostInfo = ce.TokenCostInfo(input_price=0.30,
                                      output_price=2.5,
                                      pricing_unit=1000000.0)

    GeminiRegistry = ce.CostRegistry(model=cfg.model, 
                                     info=GeminiCostInfo,
                                     registry=ce.RegistryInfo(model=cfg.model))    

    register_prompt(SYSTEM_INSTRUCTIONS, cfg.model, str(datetime.now()))

    client = build_client(cfg)

    print("Google genai.Client built successfully")

    out_dir: Optional[Path] = Path(pipe_config.out_dir).expanduser().resolve() if pipe_config.out_dir else None

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output path is = {out_dir}")
    for inp in pipe_config.input_files:
        print(f"Processing document = {fi.get_file_name(inp)}")
        input_path = Path(inp).expanduser().resolve()
        if not input_path.exists():
            print(f"ERROR: Not found: {input_path}", file=sys.stderr)
            continue

        process_input(
            client=client,
            pipe_config=pipe_config,
            input_path=input_path,
            registry_cls=GeminiRegistry
        )

    return 0


if __name__ == "__main__":
    main()
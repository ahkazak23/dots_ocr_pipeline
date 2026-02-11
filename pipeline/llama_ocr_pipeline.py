import os
import time
import json
import argparse
import pypdf
from pdf2image import convert_from_path
from PIL import ImageDraw, ImageFont
from llama_parse import LlamaParse
from llama_cloud import ParserLanguages
from llama_cloud.client import AsyncLlamaCloud
from llama_cloud.types import ClassifierRule, ClassifyParsingConfiguration
from llama_cloud_services.beta.classifier.client import LlamaClassify  # helper wrapper


# function to map command line argument language
# to LlamaParse language enum
def map_lang_code_to_enum(detected_code):
    if not detected_code:
        return ParserLanguages.EN

    detected_code = detected_code.lower()
    # mappings from langdetect to LlamaParse
    mapping = {
        # Common Abbreviations
        "en": ParserLanguages.EN,
        "eng": ParserLanguages.EN,
        "english": ParserLanguages.EN,
        "zh": ParserLanguages.CH_SIM,
        "zh-en": ParserLanguages.CH_SIM,
        "zh-cn": ParserLanguages.CH_SIM,
        "zh-tw": ParserLanguages.CH_TRA,
        "chi_sim": ParserLanguages.CH_SIM,
        "chi_tra": ParserLanguages.CH_TRA,
        "chinese": ParserLanguages.CH_SIM,
        "fr": ParserLanguages.FR,
        "french": ParserLanguages.FR,
        "de": ParserLanguages.DE,
        "german": ParserLanguages.DE,
        "ja": ParserLanguages.JA,
        "japanese": ParserLanguages.JA,
        "ko": ParserLanguages.KO,
        "korean": ParserLanguages.KO,
        "es": ParserLanguages.ES,
        "spanish": ParserLanguages.ES,
        "it": ParserLanguages.IT,
        "italian": ParserLanguages.IT,
        "ru": ParserLanguages.RU,
        "russian": ParserLanguages.RU,
        "pt": ParserLanguages.PT,
        "portuguese": ParserLanguages.PT,
        "vi": ParserLanguages.VI,
        "vietnamese": ParserLanguages.VI,
    }

    return mapping.get(detected_code, ParserLanguages.EN)


# function to create CLI command parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="LlamaParse v2 OCR evaluation pipeline"
    )

    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=None)
    parser.add_argument("--pages", type=str, default=None)
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument(
        "--draw-layout",
        action="store_true",
        default=False,
        help="Draw layout boxes on images",
    )

    parser.add_argument(
        "--denormalize-bbox",
        action="store_true",
        default=False,
        help="Whether to denormalize bboxes or not",
    )

    parser.add_argument(
        "--max-files", type=int, default=None, help="Maximum number of files to process"
    )

    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language of the documents (e.g., 'en', 'zh', 'fr'). Defaults to 'en'.",
    )

    return parser.parse_args()


# function to classify image content layout
# either as "vertical_content_text" or "standard"
def classify_content_layout(img_path, language_enum=ParserLanguages.EN):
    client = AsyncLlamaCloud(token=os.environ["LLAMA_CLOUD_API_KEY"])
    classifier = LlamaClassify(
        client, project_id="cc3df952-9a3c-43e7-afd0-7d6a12414cb5"
    )

    rules = [
        ClassifierRule(
            type="vertical_context_text",
            description="Pages where the majority of content is oriented vertically,  if tables, text, headers, footers, etc. are vertical, basically if document is not in top-to-bottom human reading order",
        ),
        ClassifierRule(
            type="standard",
            description="Pages where text is oriented horizontally in standard reading order",
        ),
    ]

    parser_config = ClassifyParsingConfiguration(
        lang=language_enum,  # language of page to be parsed
        max_pages=1,  # number of pages to parse
    )

    # note: for async usage, use `await classifier.aclassify(...)`
    results = classifier.classify(
        rules=rules, files=[img_path], parsing_configuration=parser_config
    )

    print(results.items[0])
    print(f"Processed page: {results.items[0].file.name}")

    print(
        f"Confidence for content layout detection(vertical/standard): {results.items[0].result.confidence}"
    )

    print(f"Content layout type: {results.items[0].result.type}")
    layout_content_type = results.items[0].result.type

    return layout_content_type


# utility function to parse --pages CLI argument
# into list of page numbers
def parse_pages_arg(pages_arg):
    if not pages_arg:
        return None

    pages = set()
    for p in pages_arg.split(","):
        p = p.strip()
        if not p.isdigit():
            raise ValueError(f"Invalid page number: {p}")

        pages.add(int(p))

    return sorted(pages)


# utility function to get the page number of PDF file
def get_pdf_page_count(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)

        return len(reader.pages)


# function to convert PDF page to image
def convert_pdf_page_to_image(
    pdf_file_path, page_number, raw_img_dir, dpi=300, language_enum=ParserLanguages.EN
):
    images = convert_from_path(
        pdf_path=pdf_file_path, dpi=dpi, first_page=page_number, last_page=page_number
    )

    image = images[0]
    os.makedirs(raw_img_dir, exist_ok=True)
    raw_img_path = os.path.join(raw_img_dir, f"page_{page_number}.jpg")
    image.save(raw_img_path, format="JPEG")
    layout_content_type = classify_content_layout(
        raw_img_path, language_enum=language_enum
    )

    return image, layout_content_type


# function to denormalize the bounding box coordinates
def denormalize_bbox(bbox, page_width, page_height):
    x1, y1, x2, y2 = bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]
    return [
        int(x1 * page_width),
        int(y1 * page_height),
        int(x2 * page_width),
        int(y2 * page_height),
    ]


# function to rotate image by 90 degrees clockwise
# if image layout content is vertical
def rotate_image_if_content_layout_is_vertical(image, layout_content_type):
    if layout_content_type == "vertical_context_text":
        return image.rotate(-90, expand=True)

    return image


# function draw layout boxes around detected regions
def draw_layout_boxes_on_image(image, w, h, layout_regions):
    draw = ImageDraw.Draw(image)

    # Load a larger bold font for labels
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64
        )
    except OSError:
        font = ImageFont.load_default(font_size=64)

    for region in layout_regions:
        bbox = region.get("bbox")
        if not bbox:
            continue

        px_box = denormalize_bbox(bbox, w, h)
        label = region.get("label") or "unknown"

        # for detecting other layout sections
        # more entry types can be added here with corresponding colors
        color = {
            "text": "red",
            "table": "blue",
            "sectionHeader": "green",
            "pageFooter": "orange",
            "picture": "purple",
            "keyValueRegion": "brown",
            "caption": "pink",
            "heading": "cyan",
            "form": "yellow",
            "listItem": "gold",
            "list": "gold",
        }.get(label, "gray")

        draw.rectangle(px_box, outline=color, width=8)
        draw.text((px_box[0] + 15, px_box[1] + 45), label + " ", fill=color, font=font)

    return image


# function to save annotated page to output directory
def save_annotated_page(
    pdf_path,
    page_number,
    layout_regions,
    output_dir,
    raw_img_dir,
    dpi=300,
    language_enum=ParserLanguages.EN,
):
    os.makedirs(output_dir, exist_ok=True)
    image, layout_content_type = convert_pdf_page_to_image(
        pdf_path, page_number, raw_img_dir, dpi, language_enum=language_enum
    )

    w, h = image.size
    # rotate image if content layout is "vertical"
    annotated_image = rotate_image_if_content_layout_is_vertical(
        image, layout_content_type
    )

    annotated_image = draw_layout_boxes_on_image(annotated_image, w, h, layout_regions)

    output_path = os.path.join(output_dir, f"page_{page_number}_annotated_layout.jpg")

    annotated_image.save(output_path, format="JPEG")

    return layout_content_type


# function to perform layout analysis and save annotated images
def perform_layout_analysis(cmd_args, pdf_path, results, base_name, language_enum):
    print("Drawing layout images...")
    layout_image_dir = os.path.join(
        cmd_args.output_dir, "layout_images", base_name[: base_name.rfind(".")]
    )

    raw_image_dir = os.path.join(
        cmd_args.output_dir, "raw_images", base_name[: base_name.rfind(".")]
    )

    for page_result in results:
        page_num = page_result.get("page")
        layout_regions = page_result.get("layout")

        layout_content_type = save_annotated_page(
            pdf_path=pdf_path,
            page_number=page_num,
            layout_regions=layout_regions,
            output_dir=layout_image_dir,
            raw_img_dir=raw_image_dir,
            dpi=300,
            language_enum=language_enum,
        )

        page_result["layout_orientation"] = layout_content_type


# function to parse documents with Llamaparse v2
def apply_llamaparse_v2(
    pdf_file_path,
    start_page,
    end_page,
    pages_list=None,
    language_enum=ParserLanguages.EN,
):
    print(
        f"Parsing '{os.path.basename(pdf_file_path)}' with Language Mode: {language_enum.value}"
    )

    print("PDF file path: ", pdf_file_path)
    total_pages = get_pdf_page_count(pdf_file_path)
    last_page = min(total_pages, end_page) if end_page else total_pages

    results = []

    # determine which pages to keep
    if pages_list:
        pages_to_keep = set(pages_list)
    else:
        pages_to_keep = set(range(start_page, last_page + 1))

    target_pages_str = ",".join(
        [str(page_number - 1) for page_number in sorted(pages_to_keep)]
    )

    print(f"Requesting pages (0-indexed): {target_pages_str}")

    parser = LlamaParse(
        api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),
        language=language_enum,
        result_type="markdown",
        extract_layout=True,
        split_by_page=True,
        target_pages=target_pages_str,
        verbose=True,
        show_progress=True,
    )

    start_time = time.perf_counter()
    documents = parser.get_json_result(pdf_file_path)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    # uncomment if you want to see full document output
    # print(documents)
    doc = documents[0]  # get the result for a single file

    pages = doc.get("pages")

    per_page_time = total_time / max(1, len(pages_to_keep))

    if pages:
        for page in pages:
            page_num = page.get("page")

            results.append(
                {
                    "page": page_num,
                    "time_seconds": per_page_time,
                    "items": page.get("items"),
                    "layout": page.get("layout"),
                }
            )

    return results, total_time


# function to convert Markdown table to parsed rows and columns
def parse_markdown_to_json(md_text):
    """
    Parses markdown table string into {'columns': [], 'rows': []}
    """
    if not md_text:
        return None

    lines = [line.strip() for line in md_text.strip().split("\n") if line.strip()]
    if len(lines) < 2:
        return None

    def split_row(row_str):
        # split by pipe, strip whitespace
        cells = [c.strip() for c in row_str.split("|")]
        # remove empty strings at start/end often caused by | at boundaries
        if len(cells) > 0 and cells[0] == "":
            cells.pop(0)
        if len(cells) > 0 and cells[-1] == "":
            cells.pop()

        return cells

    headers = []
    rows = []
    # try to identify header and body
    # Standard MD Table: Header \n Seperator (---) \n Data
    if len(lines) > 1 and set(
        lines[1].replace("|", "").replace(" ", "").replace(":", "")
    ) <= {"-"}:
        headers = split_row(lines[0])
        start_data_idx = 2
    else:
        # No standard seperator, treat all as rows or first as header (ambiguous we assume header)
        headers = []
        start_data_idx = 0

    for i in range(start_data_idx, len(lines)):
        rows.append(split_row(lines[i]))

    return {"columns": headers, "rows": rows}


# function to convert llamaparse outputs to json
def convert_llamaparse_output_to_json(
    results, file_name, output_dir, pdf_path, cmd_arg
):

    SPEC_TYPE_MAP = {
        "heading": "page_header",
        "sectionHeader": "section_header",
        "text": "text_block",
        "table": "table",
        "picture": "image",
        "caption": "caption",
        "pageFooter": "page_footer",
        "pageHeader": "page_header",
        "form": "form",
        "keyValueRegion": "paragraph",
        "listItem": "paragraph",
        "list": "paragraph",
        "formula": "formula",  # Could be 'image' or 'diagram' depending on use case
    }

    file_level_warnings = []
    try:
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            box = reader.pages[0].mediabox

            page_w, page_h = float(box.width), float(box.height)
            print(f"File: {file_name}, width: {page_w}, height: {page_h}")

    except Exception as e:
        print(f"Warning: Could not determine page dimensions for {file_name}: {e}")
        page_w, page_h = 0.0, 0.0
        file_level_warnings.append(f"Page dimensions could not be determined: {e}")

    # LSExtractor format requires a list of documents or single objects per page
    for page_result in results:
        current_page_warnings = list(file_level_warnings)
        blocks = []
        source_list = page_result.get("items")

        if source_list is None:
            current_page_warnings.append(
                "LlamaParse returned no 'items' for this page'."
            )
            source_list = []

        for idx, item in enumerate(source_list):
            raw_bbox = item.get("bBox")
            # 1. Normalize BBox to [x1, y1, x2, y2] scale 0-1 as per spec
            if raw_bbox:
                x1, y1 = raw_bbox["x"], raw_bbox["y"]
                x2, y2 = x1 + raw_bbox["w"], y1 + raw_bbox["h"]

                # normalization check
                if page_w > 0 and page_h > 0:
                    # If coordinates are absolute pixels (larger than 1.0)
                    if x2 > 1.0 and y2 > 1.0:
                        x1, x2 = x1 / page_w, x2 / page_w
                        y1, y2 = y1 / page_h, y2 / page_h
                        normalized_bbox = [x1, y1, x2, y2]
                else:
                    # if dimensions are missing
                    normalized_bbox = [x1, y1, x2, y2]
                    current_page_warnings.append(
                        f"Item {idx}: Normalization skipped (dimensions unknown)."
                    )

            else:
                # warning: missing bbox
                normalized_bbox = [0.0, 0.0, 0.0, 0.0]
                current_page_warnings.append(
                    f"Item index {idx}: Missing bounding box (bBox)."
                )

            # 2. Map types to spec (paragraph, table, image, diagram)
            # LlamaParse types: 'text' -> 'paragraph', 'table' -> 'table'
            llamaparse_type = item.get("type")
            mapped_type = SPEC_TYPE_MAP.get(llamaparse_type)

            if mapped_type is None:
                current_page_warnings.append(
                    f"Item index {idx}: Unknown LlamaParse type '{llamaparse_type}', defaulting to 'paragraph'."
                )
                mapped_type = "paragraph"

            # parse table data if applicable
            parsed_data = None
            content_text = item.get("md", "")

            if mapped_type == "table":
                parsed_data = parse_markdown_to_json(content_text)

            blocks.append(
                {
                    "type": mapped_type,
                    "bbox": normalized_bbox,
                    "extraction_origin": "llamaparse-v2",
                    "extraction_response": item.get("md", ""),
                    "extraction_response_parsed": {
                        "data": parsed_data,
                        "text": item.get("value", ""),
                    },
                }
            )

        output_data = {
            "document_id": file_name,
            "page_id": str(page_result.get("page")),
            "page_resolution": [int(page_w), int(page_h)],
            "warnings": current_page_warnings,
            "ocr": {"blocks": blocks},
        }

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"benchmark_{file_name}_page_{page_result.get('page')}.json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"Saved LSExtractor formatted JSON to: {output_path}")


# function to run llamaparse evaluation pipeline
def run_llamaparse_eval_pipeline():
    LLAMA_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY")
    if not LLAMA_API_KEY:
        raise RuntimeError("LLAMA_CLOUD_API_KEY is not set")

    cmd_args = parse_args()

    if not os.path.isdir(cmd_args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {cmd_args.input_dir}")

    if cmd_args.pages and (cmd_args.start_page != 1 or cmd_args.end_page is not None):
        raise ValueError("Use either --pages OR --start-page/--end-page, not both")

    pages_list = parse_pages_arg(cmd_args.pages)

    files_to_process = []
    for file in os.listdir(cmd_args.input_dir):
        if file.lower().endswith(".pdf"):
            files_to_process.append(file)

    files_to_process.sort()
    if cmd_args.max_files is not None:
        print(f"Applying limit: Processing only first {cmd_args.max_files} files.")
        files_to_process = files_to_process[: cmd_args.max_files]

    target_language = map_lang_code_to_enum(cmd_args.language)
    overall_start_time = time.perf_counter()
    if cmd_args.input_file is None:
        for file in files_to_process:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(cmd_args.input_dir, file)
                print("Processing: ", file)

                results, total_time = apply_llamaparse_v2(
                    pdf_path,
                    cmd_args.start_page,
                    cmd_args.end_page,
                    pages_list,
                    language_enum=target_language,
                )

                base_name = os.path.basename(file)
                # only perform layout drawing if command line arg is set
                if cmd_args.draw_layout:
                    perform_layout_analysis(
                        cmd_args,
                        pdf_path,
                        results,
                        base_name,
                        language_enum=target_language,
                    )

                convert_llamaparse_output_to_json(
                    results,
                    base_name[: base_name.rfind(".")],
                    cmd_args.output_dir,
                    pdf_path,
                    cmd_args,
                )

                print(f"LlamaParse time: {total_time:.3f}")

    else:
        pdf_path = (
            cmd_args.input_file
            if os.path.isabs(cmd_args.input_file)
            else os.path.join(cmd_args.input_dir, cmd_args.input_file)
        )

        results, total_time = apply_llamaparse_v2(
            pdf_path,
            cmd_args.start_page,
            cmd_args.end_page,
            pages_list,
            language_enum=target_language,
        )

        base_name = os.path.basename(cmd_args.input_file)
        # only perform layout drawing if command line arg is set
        if cmd_args.draw_layout:
            perform_layout_analysis(
                cmd_args, pdf_path, results, base_name, language_enum=target_language
            )

        convert_llamaparse_output_to_json(
            results,
            base_name[: base_name.rfind(".")],
            cmd_args.output_dir,
            pdf_path,
            cmd_args,
        )

        print(f"LlamaParse time: {total_time:.3f}")

    overall_end_time = time.perf_counter()
    print(
        f"OVERALL LlamaParse EXECUTION TIME: "
        f"{overall_end_time - overall_start_time:.3f}s"
    )


if __name__ == "__main__":
    run_llamaparse_eval_pipeline()

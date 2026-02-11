import os
import time
import json
import argparse
import pypdf
import pytesseract
from pdf2image import convert_from_path

# utility function to add command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Tesseract OCR evaluation pipeline"
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing PDF documents"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to store OCR results"
    )

    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Start page number (1-based)"
    )

    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="End page number (inclusive)"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI value for OCR"
    )

    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Comma-seperated list of pages to OCR (e.g. 5,8, 11)"
    )

    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to single PDF file to apply OCR"        
    )

    return parser.parse_args()

# utility function to parse arguments to list of page numbers
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

# utility function to get number of pages in a PDF file
def get_pdf_page_count(pdf_path):
    """Returns the number of pages in a PDF file."""
    try:
        with open(pdf_path, "rb") as f:
            pdf_reader = pypdf.PdfReader(f)
            return len(pdf_reader.pages)
    except Exception as e:
        print(f"  [!] Could not read page count for {pdf_path}: {e}")
        raise FileNotFoundError(f"Given file not found on path: {pdf_path}")

# function to apply OCR on given PDF file
def apply_ocr_on_image_with_tesseract(pdf_file_path, start_page, end_page, dpi, pages_list=None):
    # convert PDF file to image as tesseract
    # does not support PDF files as an input format
    
    num_of_pages = get_pdf_page_count(pdf_file_path)
    print("Number of pages is: ", num_of_pages)
    
    if pages_list:
        pages_to_process = [page for page in pages_list if 1 <= page <= num_of_pages]
    else:
        last_page = end_page if end_page else num_of_pages 
        pages_to_process = list(range(start_page, min(last_page, num_of_pages)+1))

    results = []

    image_output_dir = os.path.join("images", os.path.splitext(os.path.basename(pdf_file_path))[0])
    os.makedirs(image_output_dir, exist_ok=True)

    for page_num in pages_to_process:
    
        images = convert_from_path(pdf_path=pdf_file_path, 
                              first_page=page_num,
                              last_page=page_num,
                              dpi=dpi, 
                              fmt="jpeg",
                              output_folder=image_output_dir)
    
        start_time = time.perf_counter()
        image = images[0]

        image_data = pytesseract.image_to_data(
            image,   
            output_type=pytesseract.Output.DICT
        )

        end_time = time.perf_counter()

        results.append({
            "page": page_num,
            "time_seconds": end_time - start_time,
            "ocr": image_data
        })

    return results

# function to unify words that belong to 
# same (block_num, line_num) pair into a sentence
def unify_words_to_sentences(all_pages):
    unified_pages = []
    
    for page_contents in all_pages:
        page_dict = {}
        page_dict["page"] = page_contents["page"]
        page_dict["time_seconds"] = page_contents["time_seconds"]
        
        blocks, bboxes, confidence_scores = [], [], []
        unified_block_entry = {}
        unified_block_line_num_lst = []
        # only True when first block is being processed, False otherwise
        first_block = True 

        for page_blocks in page_contents["blocks"]:
            if (page_blocks["block_num"], page_blocks["line_num"]) not in unified_block_line_num_lst:
                # append unique (block_num, line_num) pair to unified_block_line_num list
                unified_block_line_num_lst.append((page_blocks["block_num"], page_blocks["line_num"]))

                if not first_block:
                    x1_lst = [bbox[0] for bbox in bboxes]
                    y1_lst = [bbox[1] for bbox in bboxes]
                    x2_lst = [bbox[2] for bbox in bboxes]
                    y2_lst = [bbox[3] for bbox in bboxes]
    
                    # computing unified bbox coordinates
                    bbox = [min(x1_lst), min(y1_lst), max(x2_lst), max(y2_lst)]
                    unified_block_entry["bbox"] = bbox  
                    unified_block_entry["confidence"] = sum(confidence_scores) / len(confidence_scores)

                    bboxes.clear()
                    confidence_scores.clear()

                    blocks.append(unified_block_entry)
                    unified_block_entry = {}
                
                else:
                    first_block = False
                
                unified_block_entry["block_num"] = page_blocks["block_num"]
                unified_block_entry["line_num"] = page_blocks["line_num"]
                unified_block_entry["text"] = page_blocks["text"]
                
                bboxes.append(page_blocks["bbox"])
                confidence_scores.append(page_blocks["confidence"])
        
            else:
                unified_block_entry["text"] += " " + page_blocks["text"]
                bboxes.append(page_blocks["bbox"])
                confidence_scores.append(page_blocks["confidence"])    

        page_dict["blocks"] = blocks
        unified_pages.append(page_dict)
    
    return unified_pages

# function to convert the image data into structured JSON format
def convert_tesseract_output_to_json(results, file_name, output_dir):
    all_pages = []
    
    for page_result in results:
        blocks = []
        image_data = page_result["ocr"]
        time_seconds = page_result["time_seconds"]
        page_num = page_result["page"]

        for i, text in enumerate(image_data["text"]):
            # if text field is not empty extract it
            if text.strip():
                blocks.append({
                    "text": text,
                    "page_num": page_num,
                    "block_num": image_data["block_num"][i],
                    "line_num": image_data["line_num"][i],
                    "bbox": [
                        image_data["left"][i],
                        image_data["top"][i],
                        image_data["left"][i] + image_data["width"][i],
                        image_data["top"][i] + image_data["height"][i]
                    ],
                    "confidence": float(image_data["conf"][i]),
                })

        all_pages.append({
            "page": page_num, 
            "time_seconds": time_seconds,
            "blocks": blocks
        })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"benchmark_{file_name}.json") 

    all_pages = unify_words_to_sentences(all_pages)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(all_pages, output_file, indent=4)

# function to run tesseract OCR pipeline
def run_tesseract_eval_pipeline():
    cmd_args = parse_args()

    # if cmd_args.input_file and (cmd_args.input_dir is not None):

    if not os.path.isdir(cmd_args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {cmd_args.input_dir}")

    if cmd_args.pages and (cmd_args.start_page != 1 or cmd_args.end_page is not None):
        raise ValueError("Use either --pages OR --start-page/--end-page, not both")
    pages_list = parse_pages_arg(cmd_args.pages)

    if cmd_args.input_file is None:
        overall_start_time = time.perf_counter()

        for file in os.listdir(cmd_args.input_dir):
            if file.lower().endswith(".pdf"):
                print("PDF file currently being processed: ", file)
                pdf_file_path = os.path.join(cmd_args.input_dir, file)
                
                # apply OCR with tesseract on given PDF file
                ocr_results = apply_ocr_on_image_with_tesseract(pdf_file_path,
                                                                cmd_args.start_page,
                                                                cmd_args.end_page,
                                                                cmd_args.dpi,
                                                                pages_list)
                # convert tesseract OCR output to JSON format
                convert_tesseract_output_to_json(ocr_results,
                                                file,
                                                cmd_args.output_dir)
        
        overall_end_time = time.perf_counter()
        execution_time = overall_end_time - overall_start_time
        print(f"OVERALL TESSERACT OCR EXECUTION TIME: {execution_time:.3f}")
    
    else: 
        overall_start_time = time.perf_counter()
        print("PDF file currently being processed: ", cmd_args.input_file)
        pdf_file_path = os.path.join(cmd_args.input_dir, cmd_args.input_file)
        
        # apply OCR with tesseract on given PDF file
        ocr_results = apply_ocr_on_image_with_tesseract(pdf_file_path,
                                                        cmd_args.start_page,
                                                        cmd_args.end_page,
                                                        cmd_args.dpi,
                                                        pages_list)
        # convert tesseract OCR output to JSON format
        convert_tesseract_output_to_json(ocr_results,
                                        cmd_args.input_file,
                                        cmd_args.output_dir)

        overall_end_time = time.perf_counter()
        execution_time = overall_end_time - overall_start_time

        print(f"TESSERACT OCR EXECUTION TIME for file {cmd_args.input_file}: {execution_time:.3f}") 
   

if __name__ == "__main__":
    run_tesseract_eval_pipeline()
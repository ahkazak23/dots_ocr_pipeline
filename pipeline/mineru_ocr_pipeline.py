import time
import json
import subprocess
import pypdf
import os
import argparse
from pathlib import Path


# --- DEFAULT CONFIGURATION ----
DEFAULT_INPUT_DIR = "./documents"
DEFAULT_OUTPUT_DIR = "./mineru_results"
# mode to use with MinerU processing pipeline
DEFAULT_METHOD = "ocr"
# ------------------------------


# function to create command line parser with
# command line arguments to be provided to MinerU pipeline
def parse_cmd_args():
    parser = argparse.ArgumentParser(description="MinerU PDF Extraction Pipeline")

    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing source PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save MinerU files.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Optional: Process a single PDF file instead of a directory.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default=DEFAULT_METHOD,
        choices=["ocr", "auto"],
        help="MinerU processing method ('ocr', or 'auto').",
    )

    parser.add_argument(
        "--start-page",
        type=int,
        default=0,
        help="Start page index (0-based). Default: 0",
    )

    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="End page indexx (0-based). Default: All pages",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit the number of files processed",
    )

    return parser.parse_args()


def get_pdf_page_count(pdf_path):
    """Returns the number of pages in a PDF file."""
    try:
        with open(pdf_path, "rb") as f:
            pdf_reader = pypdf.PdfReader(f)
            return len(pdf_reader.pages)
    except Exception as e:
        print(f"  [!] Could not read page count for {pdf_path}: {e}")
        return None


# function to get PDF file name
def get_pdf_name(pdf_path_obj):
    print(pdf_path_obj.name)
    return pdf_path_obj.name


def run_evaluation():
    cmd_args = parse_cmd_args()

    input_path = Path(cmd_args.input_dir)
    output_path = Path(cmd_args.output_dir)

    # ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = []
    if cmd_args.input_file:
        # single file proccessing mode
        single_file = Path(cmd_args.input_file)
        if not single_file.is_absolute():
            single_file = input_path / single_file

        if single_file.exists():
            pdf_files.append(single_file)
        else:
            print(f"Error: Input file not found: {single_file}")
            return
    else:
        # batch directory mode
        if not input_path.exists():
            print(f"Error: Input directory not found: {input_path}")
            return

        pdf_files = list(input_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found")
        return

    # apply max-files limit if requested
    if cmd_args.max_files:
        print(f"Applying limit: Processing first {cmd_args.max_files}.")
        pdf_files = pdf_files[: cmd_args.max_files]

    pdf_files.sort(key=get_pdf_name)
    records = []
    page_range_str = (
        f"Pages {cmd_args.start_page}-{cmd_args.end_page}"
        if cmd_args.end_page is not None
        else "All Pages"
    )

    print(f"\n--- MinerU Pipeline Configuration ---")
    print(f"Files: {len(pdf_files)}")
    print(f"Method: {cmd_args.method}")
    print(f"Output: {output_path}")
    print(f"Range: {page_range_str}")
    print(f"---------------------------------------")

    for pdf in pdf_files:
        print(f"Processing: {pdf.name}")

        # dynamically calculate page range
        total_num_of_pages = get_pdf_page_count(pdf)
        current_end_page = cmd_args.end_page

        # adjust ending page number logic
        if total_num_of_pages is not None:
            # if user provided an end_page
            if cmd_args.end_page is not None:
                # clamp it to the actual file length
                if cmd_args.end_page >= total_num_of_pages:
                    current_end_page = total_num_of_pages - 1
                    print(
                        f"    -> File has {total_num_of_pages} pages. Clamping end_page to {current_end_page}."
                    )
                else:
                    print(
                        f"    -> Using configured range: {cmd_args.start_page} to {current_end_page}"
                    )

            else:
                # no limit set, no need to pass -e argument to MinerU (it defaults to end)
                pass

        # construct MinerU CLI command
        # syntaxt mineru -p <input_path> -o <output_path> -m <method>
        cmd = [
            "mineru",
            "-p",
            str(pdf),
            "-o",
            str(output_path),
            "-m",
            cmd_args.method,
            "-b",
            "pipeline",  # 'pipeline' is standard CPU-friendly backend
            "-d",
            "cpu",  # Force CPU (change to 'cuda' if you have GPU support setup)
        ]

        # append page limits if configured
        if cmd_args.start_page is not None:
            cmd.extend(["-s", str(cmd_args.start_page)])

        if current_end_page is not None:
            cmd.extend(["-e", str(current_end_page)])

        # start processing
        start_time = time.perf_counter()

        result = ""
        status = "Unknown"
        error_msg = ""

        try:
            # capture output to avoid spamming console, console check=True raises on non-zero exit
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            status = "Success"

        except subprocess.CalledProcessError as e:
            status = "Failed"
            error_msg = e.stderr
            print(f"    [!] MinerU Failed: {e.stderr[:300]}...")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # verify output file existence
        # MinerU output structure: output_dir / <pdf_name_no_ext> / <pdf_name_no_ext>_content_list.json
        output_subfolder = output_path / pdf.stem
        expected_json = output_subfolder / cmd_args.method / f"{pdf.stem}_model.json"
        print(f"Expected JSON file is: {expected_json}.")

        json_found = False
        if expected_json.exists():
            json_found = True
        else:
            if status == "Success":
                print(
                    f"    [!] WARNING: Process finished but output JSON not found at {expected_json}"
                )
                status = "Missing Output"

        # log results
        record = {
            "filename": pdf.name,
            "status": status,
            "config_method": cmd_args.method,
            "page_range": (
                f"{cmd_args.start_page}-{current_end_page}"
                if current_end_page
                else "All"
            ),
            "processing_time_seconds": round(elapsed_time, 4),
            "output_json_path": str(expected_json) if json_found else None,
            "error_log": error_msg,
        }

        records.append(record)
        print(f"    -> {status} in {elapsed_time:.2f}s")

    # save benchmark report
    report_path = os.path.join(cmd_args.output_dir, "mineru_benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)

    print(f"\nPipeline complete. Report saved to '{report_path}'")


if __name__ == "__main__":
    run_evaluation()

"""
Utility Bill Decoder - Hackathon Level 1 & Level 2
- Level 1: Extracts structured JSON data from utility bill images.
- Level 2: Answers natural language questions on utility bills using visual QA.
Progressively saves outputs to CSV format with robust error and rate limit handling.
"""

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import pandas as pd

# -----------------------------------------------------------------------------
# Path Configurations
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Default directories conforming to project structure:
# train/bills/ (contains images), output/ (for CSVs), test/csv/ (for test queries)
DEFAULT_INPUT_DIR = PROJECT_ROOT / "train" / "bills"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_LEVEL1_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "level1.csv"

DEFAULT_LEVEL2_INPUT_DIR = PROJECT_ROOT / "test" / "csv"
DEFAULT_LEVEL2_INPUT_CSV = DEFAULT_LEVEL2_INPUT_DIR / "level2.csv"
DEFAULT_LEVEL2_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "level2.csv"

# Supported image file extensions
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}

# -----------------------------------------------------------------------------
# Level 1 Schema and Prompts
# -----------------------------------------------------------------------------
LEVEL1_SCHEMA_KEYS = [
    "provider",
    "tariff",
    "sanctioned_load_kw",
    "bill_month",
    "reading_date",
    "issue_date",
    "due_date",
    "previous_reading",
    "current_reading",
    "units_consumed",
    "charges",
    "total_charges",
    "taxes",
    "total_taxes",
    "current_bill",
    "arrears",
    "payable_within_due_date",
    "payable_after_due_date",
]

LEVEL1_SYSTEM_PROMPT = """You are an expert AI data extraction engine specialized in Pakistani utility bills.
Your task is to analyze the provided utility bill image and extract all billing details into a strict JSON object.

You MUST extract the data conforming EXACTLY to the following schema:
{
  "provider": "KE" | "LESCO" | "IESCO",
  "tariff": string or null,
  "sanctioned_load_kw": float or integer or null,
  "bill_month": "YYYY-MM" (e.g., "2024-05") or null,
  "reading_date": "YYYY-MM-DD" or null,
  "issue_date": "YYYY-MM-DD" or null,
  "due_date": "YYYY-MM-DD" or null,
  "previous_reading": float or integer or null,
  "current_reading": float or integer or null,
  "units_consumed": float or integer or null,
  "charges": [
    {
      "type": string,
      "amount": float or integer
    }
  ],
  "total_charges": float or integer or null,
  "taxes": [
    {
      "type": string,
      "amount": float or integer
    }
  ],
  "total_taxes": float or integer or null,
  "current_bill": float or integer or null,
  "arrears": float or integer or null,
  "payable_within_due_date": float or integer or null,
  "payable_after_due_date": float or integer or null
}

Strict Rules:
1. "provider": Identify the electric utility provider. Must be EXACTLY one of: "KE", "LESCO", or "IESCO".
   - KE = K-Electric (Karachi)
   - LESCO = Lahore Electric Supply Company
   - IESCO = Islamabad Electric Supply Company
2. "bill_month": Format strictly as "YYYY-MM" (e.g. "2023-08"). If only month/year is given (e.g. "AUG 23"), normalize to YYYY-MM.
3. "reading_date", "issue_date", "due_date": Format strictly as "YYYY-MM-DD".
4. "charges": List of all base electric tariff charges/breakdowns (e.g., Variable Charges, Fuel Price Adjustment/FPA, FC Surcharge, Tariff Differential, Quarter Tariff Adjustment/QTA).
5. "taxes": List of all taxes/duties/fees (e.g., GST, Electricity Duty, TV Fee, Sales Tax, Further Tax, Withholding Tax).
6. "charges" and "taxes" must be lists of objects with keys "type" (string) and "amount" (number). If none found, provide an empty list [].
7. Numeric fields: Extract as raw numbers (int or float). Do NOT include commas, currency symbols, or unit strings (e.g. 15200.50, not "Rs. 15,200.50").
8. Missing values: If any field is not visible, unreadable, or not present on the bill, set its value to null. Never omit any key from the root object.
9. Return ONLY the JSON object. Do not wrap in markdown quotes or provide conversational text.
"""

LEVEL1_USER_PROMPT = "Extract all utility bill fields from this image strictly according to the specified JSON schema."

# -----------------------------------------------------------------------------
# Level 2 Prompt Instruction
# -----------------------------------------------------------------------------
LEVEL2_INSTRUCTION = (
    "Act as an expert utility bill decoder. "
    "Answer the user's question concisely in plain English based strictly on the provided bill image. "
    "Perform math if needed, label estimates clearly, and state if information is missing."
)


# -----------------------------------------------------------------------------
# Common Utility Functions
# -----------------------------------------------------------------------------
def setup_gemini_client() -> str:
    """
    Loads Gemini API key from .env file or system environment variables
    and configures the google-generativeai SDK.
    """
    load_dotenv()
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(SCRIPT_DIR / ".env")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key or api_key.strip() in ("", "your_api_key_here"):
        print("\n" + "=" * 70, file=sys.stderr)
        print("[ERROR] Gemini API key not found or still set to placeholder!", file=sys.stderr)
        print("Please create a .env file in the project root with:", file=sys.stderr)
        print("    GEMINI_API_KEY=your_actual_api_key_here", file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

    genai.configure(api_key=api_key.strip())
    return api_key.strip()


def find_image_for_bill(bill_id: str, input_dir: Path) -> Optional[Path]:
    """
    Locates the corresponding image for a given bill_id in the input bills directory.
    Supports various extensions and case-insensitive matching.
    """
    clean_id = Path(str(bill_id).strip()).stem

    # 1. Direct match with standard extensions
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        candidate = input_dir / f"{clean_id}{ext}"
        if candidate.is_file():
            return candidate

    # 2. Case-insensitive search
    clean_id_lower = clean_id.lower()
    if input_dir.exists():
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.stem.lower() == clean_id_lower:
                if file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    return file_path

    return None


# -----------------------------------------------------------------------------
# Level 1 Helper Functions
# -----------------------------------------------------------------------------
def get_existing_processed_ids_level1(csv_path: Path) -> Set[str]:
    """Reads existing Level 1 CSV to identify already processed bill_ids."""
    processed_ids: Set[str] = set()
    if not csv_path.exists():
        return processed_ids

    try:
        with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header and len(header) >= 1:
                for row in reader:
                    if row:
                        processed_ids.add(row[0].strip())
    except Exception as e:
        print(f"[WARNING] Could not read existing {csv_path.name}: {e}. Proceeding fresh.")

    return processed_ids


def init_output_csv_level1(csv_path: Path) -> None:
    """Ensures output directory exists and writes Level 1 CSV header if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bill_id", "json"])
        print(f"[INIT] Initialized output CSV at: {csv_path}")


def clean_and_validate_json_level1(raw_text: str) -> str:
    """Validates model JSON and ensures all 18 required schema keys are present."""
    text = raw_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Extracted JSON root must be an object/dict, got {type(parsed).__name__}")

    normalized: Dict[str, Any] = {}
    for key in LEVEL1_SCHEMA_KEYS:
        if key in parsed:
            val = parsed[key]
            if key in ("charges", "taxes") and not isinstance(val, list):
                normalized[key] = []
            else:
                normalized[key] = val
        else:
            if key in ("charges", "taxes"):
                normalized[key] = []
            else:
                normalized[key] = None

    return json.dumps(normalized, ensure_ascii=False)


def process_bill_image_level1(
    model: genai.GenerativeModel,
    image_path: Path,
    max_retries: int = 3,
    retry_delay_seconds: float = 5.0,
) -> Optional[str]:
    """Sends image to Gemini for Level 1 JSON extraction with exponential backoff."""
    try:
        pil_img = Image.open(image_path)
    except Exception as img_err:
        print(f"  [ERROR] Unable to open image file {image_path.name}: {img_err}")
        return None

    contents = [
        LEVEL1_SYSTEM_PROMPT,
        LEVEL1_USER_PROMPT,
        pil_img,
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(contents)
            if not response.text:
                raise ValueError("Model returned an empty response")
            return clean_and_validate_json_level1(response.text)

        except Exception as e:
            err_msg = str(e)
            is_rate_limit = (
                "429" in err_msg
                or "ResourceExhausted" in err_msg
                or "quota" in err_msg.lower()
                or "rate limit" in err_msg.lower()
            )
            is_server_error = "500" in err_msg or "503" in err_msg or "Overloaded" in err_msg

            if (is_rate_limit or is_server_error) and attempt < max_retries:
                wait_time = retry_delay_seconds * (2 ** (attempt - 1))
                reason = "Rate limit hit" if is_rate_limit else "Server busy"
                print(f"  [{reason}] Waiting {wait_time:.1f}s before retry (attempt {attempt}/{max_retries})...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  [API ERROR] Level 1 extraction failed for {image_path.name}: {e}")
                return None

    return None


def append_result_to_csv_level1(csv_path: Path, bill_id: str, json_str: str) -> None:
    """Progressively appends Level 1 result to CSV, escaping double quotes properly."""
    with open(csv_path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([bill_id, json_str])
        f.flush()


def run_level1(args, model_name: str):
    """Executes Level 1 extraction pipeline."""
    input_dir: Path = args.input_dir.resolve()
    output_csv: Path = args.level1_output.resolve()

    print("\n" + "=" * 70)
    print("           UTILITY BILL DECODER - LEVEL 1 (JSON EXTRACTION)")
    print("=" * 70)
    print(f"Model:           {model_name}")
    print(f"Input Directory: {input_dir}")
    print(f"Output CSV:      {output_csv}")
    print("=" * 70)

    # Initialize Gemini model with forced JSON output
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json",
        temperature=0.1,
    )
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )

    if not input_dir.exists():
        print(f"\n[ERROR] Input directory does not exist: {input_dir}", file=sys.stderr)
        print("Please ensure images are placed in ../train/bills/", file=sys.stderr)
        return

    image_files: List[Path] = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )

    total_images = len(image_files)
    print(f"[INFO] Found {total_images} bill image(s) in {input_dir.name}/")

    if total_images == 0:
        print("[WARNING] No supported images found to process for Level 1.")
        return

    init_output_csv_level1(output_csv)
    processed_ids = set() if args.force else get_existing_processed_ids_level1(output_csv)

    if processed_ids:
        print(f"[INFO] Found {len(processed_ids)} already processed bill(s) in {output_csv.name}. Skipping them.")

    print("\n" + "-" * 70)
    print("Starting Level 1 bill extraction...")
    print("-" * 70)

    success_count = 0
    skip_count = 0
    error_count = 0
    start_time = time.time()

    for idx, img_path in enumerate(image_files, start=1):
        bill_id = img_path.stem
        progress_prefix = f"[L1 {idx}/{total_images}]"

        if bill_id in processed_ids:
            print(f"{progress_prefix} [SKIP] '{bill_id}' already present in CSV.")
            skip_count += 1
            continue

        print(f"{progress_prefix} [PROCESSING] {img_path.name} (bill_id: '{bill_id}')...")
        json_output = process_bill_image_level1(model, img_path)

        if json_output is not None:
            append_result_to_csv_level1(output_csv, bill_id, json_output)
            processed_ids.add(bill_id)
            success_count += 1
            print(f"  [SUCCESS] Extracted & saved '{bill_id}' to {output_csv.name}")
        else:
            error_count += 1
            print(f"  [FAILED] Skipping '{bill_id}' due to error.")

        if idx < total_images and args.delay > 0:
            time.sleep(args.delay)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("                 LEVEL 1 PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total Images:       {total_images}")
    print(f"Successfully Saved: {success_count}")
    print(f"Skipped (Prior):    {skip_count}")
    print(f"Errors/Failed:      {error_count}")
    print(f"Time Elapsed:       {elapsed:.2f} seconds")
    print(f"Output File:        {output_csv}")
    print("=" * 70 + "\n")


# -----------------------------------------------------------------------------
# Level 2 Helper Functions & Pipeline
# -----------------------------------------------------------------------------
def ensure_level2_test_csv(test_csv_path: Path) -> Path:
    """
    Checks if ../test/csv/level2.csv exists. If not, automatically creates the
    parent directory and a dummy file with columns: bill_id, question_id, question, answer
    and adds one dummy test row: KESC_0008, Q1, "What is the total amount due?", ""
    """
    if not test_csv_path.exists():
        test_csv_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_df = pd.DataFrame(
            [
                {
                    "bill_id": "KESC_0008",
                    "question_id": "Q1",
                    "question": "What is the total amount due?",
                    "answer": "",
                }
            ]
        )
        dummy_df.to_csv(test_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"[INIT] Created test CSV template with dummy row at: {test_csv_path}")
    return test_csv_path


def answer_bill_question_level2(
    model: genai.GenerativeModel,
    image_path: Path,
    question: str,
    max_retries: int = 3,
    retry_delay_seconds: float = 5.0,
) -> Optional[str]:
    """
    Sends bill image and question to Gemini model and retrieves concise plain English answer.
    Implements retries with exponential backoff for rate limits or transient errors.
    """
    try:
        pil_img = Image.open(image_path)
    except Exception as img_err:
        print(f"  [ERROR] Unable to open image file {image_path.name}: {img_err}")
        return None

    contents = [
        LEVEL2_INSTRUCTION,
        f"Question: {question}",
        pil_img,
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(contents)
            if not response.text:
                raise ValueError("Model returned an empty response")
            return response.text.strip()

        except Exception as e:
            err_msg = str(e)
            is_rate_limit = (
                "429" in err_msg
                or "ResourceExhausted" in err_msg
                or "quota" in err_msg.lower()
                or "rate limit" in err_msg.lower()
            )
            is_server_error = "500" in err_msg or "503" in err_msg or "Overloaded" in err_msg

            if (is_rate_limit or is_server_error) and attempt < max_retries:
                wait_time = retry_delay_seconds * (2 ** (attempt - 1))
                reason = "Rate limit hit" if is_rate_limit else "Server busy"
                print(f"  [{reason}] Waiting {wait_time:.1f}s before retry (attempt {attempt}/{max_retries})...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  [API ERROR] Answering failed for question '{question}': {e}")
                return None

    return None


def run_level2(args, model_name: str):
    """Executes Level 2 Visual QA pipeline."""
    input_bills_dir: Path = args.input_dir.resolve()
    test_csv_path: Path = args.level2_input.resolve()
    output_csv_path: Path = args.level2_output.resolve()

    print("\n" + "=" * 70)
    print("           UTILITY BILL DECODER - LEVEL 2 (VISUAL QUESTION ANSWERING)")
    print("=" * 70)
    print(f"Model:           {model_name}")
    print(f"Bills Directory: {input_bills_dir}")
    print(f"Test CSV:        {test_csv_path}")
    print(f"Output CSV:      {output_csv_path}")
    print("=" * 70)

    # 1. Ensure test CSV exists, create dummy row if missing
    ensure_level2_test_csv(test_csv_path)

    # 2. Read test CSV using pandas
    try:
        df = pd.read_csv(test_csv_path, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"[ERROR] Failed to read test CSV at {test_csv_path}: {e}", file=sys.stderr)
        return

    required_cols = {"bill_id", "question_id", "question"}
    if not required_cols.issubset(df.columns):
        print(
            f"[ERROR] Test CSV must contain columns {required_cols}. Found: {list(df.columns)}",
            file=sys.stderr,
        )
        return

    if "answer" not in df.columns:
        df["answer"] = ""
    else:
        df["answer"] = df["answer"].astype(object)

    # 3. Check existing output CSV to support safe resume
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if output_csv_path.exists() and not args.force:
        try:
            existing_out_df = pd.read_csv(output_csv_path, dtype=str, keep_default_na=False)
            if {"bill_id", "question_id", "answer"}.issubset(existing_out_df.columns):
                # Map (bill_id, question_id) -> answer
                answer_map = {}
                for _, r in existing_out_df.iterrows():
                    ans = str(r["answer"]).strip()
                    if ans and ans.lower() != "nan":
                        answer_map[(str(r["bill_id"]).strip(), str(r["question_id"]).strip())] = ans

                for idx, row in df.iterrows():
                    key = (str(row["bill_id"]).strip(), str(row["question_id"]).strip())
                    if key in answer_map and not str(row["answer"]).strip():
                        df.at[idx, "answer"] = answer_map[key]
        except Exception as e:
            print(f"[WARNING] Could not load prior Level 2 output for resume: {e}")

    # 4. Initialize Gemini Model (plain English text response)
    print(f"\n[INFO] Initializing Gemini model for Q&A: {model_name}...")
    generation_config = genai.types.GenerationConfig(
        temperature=0.1,
    )
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )

    total_rows = len(df)
    print(f"[INFO] Found {total_rows} question(s) in {test_csv_path.name}")

    if total_rows == 0:
        print("[WARNING] Test CSV contains no rows. Exiting Level 2.")
        return

    print("\n" + "-" * 70)
    print("Starting Level 2 question answering...")
    print("-" * 70)

    success_count = 0
    skip_count = 0
    error_count = 0
    start_time = time.time()

    for idx, row in df.iterrows():
        bill_id = str(row["bill_id"]).strip()
        question_id = str(row["question_id"]).strip()
        question = str(row["question"]).strip()
        current_answer = str(row["answer"]).strip()

        row_num = idx + 1
        progress_prefix = f"[L2 {row_num}/{total_rows}]"

        # Check if already answered
        if current_answer and current_answer.lower() != "nan" and not args.force:
            print(f"{progress_prefix} [SKIP] '{bill_id}' ({question_id}) already answered.")
            skip_count += 1
            continue

        print(f"{progress_prefix} [PROCESSING] bill_id: '{bill_id}', question_id: '{question_id}'")
        print(f"  Question: \"{question}\"")

        # Locate corresponding bill image in train/bills/
        image_path = find_image_for_bill(bill_id, input_bills_dir)
        if not image_path:
            error_count += 1
            missing_msg = f"[MISSING IMAGE] Bill image for '{bill_id}' not found in {input_bills_dir}."
            print(f"  {missing_msg} Skipping...")
            continue

        print(f"  Image Located: {image_path.name}")

        # Ask Gemini model
        answer = answer_bill_question_level2(model, image_path, question)

        if answer is not None:
            df.at[idx, "answer"] = answer
            success_count += 1
            print(f"  [SUCCESS] Answer: {answer}")
        else:
            error_count += 1
            print(f"  [FAILED] Could not answer '{question_id}'.")

        # Progressively save updated DataFrame back to output/level2.csv with proper quoting
        df.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
        print(f"  [SAVED] Progressively updated {output_csv_path.name}")

        # Throttle between calls
        if row_num < total_rows and args.delay > 0:
            time.sleep(args.delay)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("                 LEVEL 2 PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total Questions:    {total_rows}")
    print(f"Successfully Saved: {success_count}")
    print(f"Skipped (Prior):    {skip_count}")
    print(f"Errors/Failed:      {error_count}")
    print(f"Time Elapsed:       {elapsed:.2f} seconds")
    print(f"Output File:        {output_csv_path}")
    print("=" * 70 + "\n")


# -----------------------------------------------------------------------------
# Main CLI Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Utility Bill Decoder (Level 1 & Level 2) using Google Gemini"
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["1", "2", "all"],
        default="all",
        help="Specify which level to run: '1', '2', or 'all' (default: all)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing bill images (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--level1-output",
        type=Path,
        default=DEFAULT_LEVEL1_OUTPUT_CSV,
        help=f"Output CSV path for Level 1 (default: {DEFAULT_LEVEL1_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--level2-input",
        type=Path,
        default=DEFAULT_LEVEL2_INPUT_CSV,
        help=f"Input test CSV path for Level 2 (default: {DEFAULT_LEVEL2_INPUT_CSV})",
    )
    parser.add_argument(
        "--level2-output",
        type=Path,
        default=DEFAULT_LEVEL2_OUTPUT_CSV,
        help=f"Output CSV path for Level 2 (default: {DEFAULT_LEVEL2_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        help="Gemini model name (default: gemini-3.5-flash-lite)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between successive API calls to avoid rate limits (default: 1.0)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of bills/questions already present in output CSVs",
    )

    args = parser.parse_args()
    model_name = args.model

    print("=" * 70)
    print("              UTILITY BILL DECODER - HACKATHON")
    print("=" * 70)
    print(f"Target Level:    {args.level}")
    print(f"Model:           {model_name}")
    print("=" * 70)

    # Configure Gemini API
    setup_gemini_client()

    # Run selected level(s)
    if args.level in ("1", "all"):
        run_level1(args, model_name)

    if args.level in ("2", "all"):
        run_level2(args, model_name)


if __name__ == "__main__":
    main()

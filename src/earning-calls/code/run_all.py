from pathlib import Path
import json
import argparse

from clean_transcript import load_and_clean
from pipeline import run_pipeline


def process(pdf_path: str, out_json: str) -> dict:
    full_text, qa_text, txt_path = load_and_clean(pdf_path)
    result = run_pipeline(full_text, qa_text)
    Path(out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end earnings call pipeline")
    parser.add_argument("--pdf", required=True, help="Path to the earnings call PDF")
    parser.add_argument("--out", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    output = process(args.pdf, args.out)
    print(json.dumps(output, indent=2))
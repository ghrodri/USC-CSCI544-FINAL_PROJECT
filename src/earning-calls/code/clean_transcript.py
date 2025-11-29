import re
from pathlib import Path
import pdfplumber


BASE_DIR = Path("./src/earning-calls")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def extract_pdf_text(pdf_path: str) -> str:
    pdf_path = Path(pdf_path)
    text_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_pages.append(page_text)
    return "\n".join(text_pages)


def basic_clean(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def strip_participants_and_header(text: str) -> str:
    patterns = [
        r"Company Participants.*?Presentation",
        r"Conference Call Participants.*?Presentation",
    ]
    for pat in patterns:
        text = re.sub(pat, "Presentation", text, flags=re.S | re.I)
    return text


def split_presentation_and_qa(text: str) -> tuple[str, str]:
    qa_match = re.search(
        r"(Question[-\s]and[-\s]Answer Session|Question and Answer Session|Q&A)",
        text,
        flags=re.I,
    )
    if not qa_match:
        return text.strip(), ""

    qa_start = qa_match.start()
    full_clean = text.strip()
    qa_section = text[qa_start:].strip()

    qa_lines = qa_section.splitlines()
    if qa_lines:
        qa_section = "\n".join(qa_lines[1:]).strip()

    return full_clean, qa_section


def clean_transcript_sections(raw_text: str) -> tuple[str, str]:
    text = basic_clean(raw_text)
    text = strip_participants_and_header(text)
    full_clean, qa_clean = split_presentation_and_qa(text)
    return full_clean, qa_clean


def load_and_clean(pdf_path: str) -> tuple[str, str, Path]:
    raw = extract_pdf_text(pdf_path)
    full_clean, qa_clean = clean_transcript_sections(raw)

    pdf_stem = Path(pdf_path).stem
    txt_path = BASE_DIR / f"{pdf_stem}_clean.txt"
    txt_path.write_text(full_clean, encoding="utf-8")

    return full_clean, qa_clean, txt_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    args = parser.parse_args()
    full, qa, path = load_and_clean(args.pdf)
    print(f"Saved cleaned transcript to: {path}")
    print(f"Full length: {len(full)} chars | QA length: {len(qa)} chars")
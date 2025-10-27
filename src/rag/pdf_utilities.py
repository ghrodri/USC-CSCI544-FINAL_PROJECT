import os
import re
import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_path


PDF_DIR = "pdfs/"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_path in pdf_docs:
        full_path = pdf_path
        if not os.path.isabs(full_path) and not os.path.exists(full_path):
            full_path = os.path.join(PDF_DIR, pdf_path) # For local run

        if not os.path.exists(full_path):
            print("[WARN] File not found:", full_path)
            continue

        print("[INFO] Reading:", full_path)
        extracted_text = ""
        try:
            pdf_reader = PdfReader(full_path)
            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                except Exception as e:
                    print(f"[WARN] Page {i+1} read error: {e}")
                    page_text = None
                if page_text and page_text.strip():
                    extracted_text += page_text + "\n"
                else:
                    print(f"[INFO] Using OCR for page {i+1}...")
                    extracted_text += extract_text_ocr(full_path, i)
        except Exception as e:
            print(f"[WARN] PyPDF2 failed for '{full_path}': {e}")
            print("[INFO] Running full OCR for file...")
            extracted_text = extract_text_ocr(full_path)
        text += extracted_text + "\n"
    return text.strip()


def extract_text_ocr(pdf_path, page_index=None):
    text = ""
    try:
        pages = convert_from_path(pdf_path)
        if page_index is not None:
            pages = [pages[page_index]]
        for img in pages:
            text += pytesseract.image_to_string(img, lang="eng") + "\n"
    except Exception as e:
        print(f"[ERROR] OCR failed for '{pdf_path}': {e}")
    return text


def clean_text(text):
    """
    Normalize newlines while preserving both single and double line breaks.
    - Convert CRLF to LF.
    - Collapse 3+ newlines to exactly two.
    - Trim trailing spaces and collapse multiple spaces.
    """
    text = text.replace("\r\n", "\n")
    # Preserve single newlines; only collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
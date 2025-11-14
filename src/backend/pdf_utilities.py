import os
import re
import pytesseract
import pdfplumber
from pdf2image import convert_from_path


PDF_DIR = "pdfs/"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_path in pdf_docs:
        full_path = pdf_path if os.path.isabs(pdf_path) else os.path.join(PDF_DIR, pdf_path)
        if not os.path.exists(full_path):
            print("[WARN] File not found:", full_path)
            continue
        
        print("[INFO] Reading:", full_path)
        try:
            with pdfplumber.open(full_path) as pdf:
                for page in pdf.pages:
                    # Extract tables with structure
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            # Convert to readable format
                            for row in table:
                                text += " | ".join([str(cell or "") for cell in row]) + "\n"
                            text += "\n"
                    # Then text
                    text += (page.extract_text() or "") + "\n"
        except Exception as e:
            print(f"[ERROR] pdfplumber failed: {e}")
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
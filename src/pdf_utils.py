import fitz  # PyMuPDF
import os
from typing import Optional

def extract_pdf_text(pdf_path: str) -> Optional[str]:
    """Extracts text content from a PDF file using PyMuPDF."""
    if not os.path.exists(pdf_path): print(f"  Error: PDF file not found: {pdf_path}"); return None
    try:
        doc = fitz.open(pdf_path); full_text = "".join(page.get_text("text") for page in doc); doc.close()
        if not full_text.strip(): print(f"  Warning: Extracted text from {pdf_path} is empty."); return ""
        full_text = '\n'.join(line for line in full_text.splitlines() if line.strip()); return full_text
    except fitz.fitz.FileDataError: print(f"  Error: Failed to open PDF (Invalid data): {pdf_path}"); return None
    except Exception as e: print(f"  Error extracting text from PDF {pdf_path}: {e}"); return None 
import pdfplumber
import fitz
from pathlib import Path

from .base import wrap_extractor, base_result

# Извлечение текста/таблиц/изображений из PDF (pdfplumber)
@wrap_extractor
def extract_pdf_text_plumber(path: Path):
    res = base_result()   # ← ИСПРАВЛЕНО
    res["source"] = "native_pdfplumber"

    text_parts, blocks, tables, images = [], [], [], []

    with pdfplumber.open(path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
                blocks.append({
                    "page": page_idx + 1,
                    "bbox": None,
                    "text": t.strip(),
                    "type": "text",
                })

            try:
                tbls = page.extract_tables()
                for tbl in tbls:
                    rows = [[cell or "" for cell in row] for row in tbl]
                    if rows:
                        tables.append({"page": page_idx + 1, "rows": rows})
            except Exception as e:
                res["warnings"].append(f"table error: {e}")

            for im in page.images:
                images.append({
                    "page": page_idx + 1,
                    "bbox": [im["x0"], im["top"], im["x1"], im["bottom"]],
                })

    res["text"] = "\n".join(text_parts)
    res["blocks"] = blocks
    res["tables"] = tables
    res["images"] = images
    return res

# Извлечение текста и изображений из PDF (PyMuPDF blocks)
@wrap_extractor
def extract_pdf_text_fitz(path: Path):
    res = base_result()   # ← ИСПРАВЛЕНО
    res["source"] = "native_pymupdf"

    doc = fitz.open(path)
    text_parts, blocks, images = [], [], []

    for page_idx, page in enumerate(doc):
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, text, *_ = b
            if text.strip():
                text_parts.append(text.strip())
                blocks.append({
                    "page": page_idx + 1,
                    "bbox": [x0, y0, x1, y1],
                    "text": text.strip(),
                    "type": "text",
                })

        for img in page.get_images(full=True):
            images.append({"page": page_idx + 1, "xref": img[0]})

    res["text"] = "\n".join(text_parts)
    res["blocks"] = blocks
    res["images"] = images
    return res

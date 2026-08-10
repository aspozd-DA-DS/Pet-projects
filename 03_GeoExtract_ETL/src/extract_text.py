import time
import json
import cv2
import pytesseract
from pathlib import Path

import fitz
import pdfplumber
import xlrd
import pandas as pd
from docx import Document
from striprtf.striprtf import rtf_to_text

from src.extractors.pdf_text_extractor import extract_pdf_text_fitz
from src.extractors.pdf_scan_extractor import extract_pdf_scanned
from src.extractors.docx_extractor import (
    extract_docx,
    convert_docx_to_pdf,
    convert_doc_to_docx
)
from src.extractors.excel_extractor import extract_excel
from src.extractors.image_extractor import extract_image
from src.extractors.txt_extractor import extract_txt
from src.extractors.rtf_extractor import extract_rtf
from src.extractors.base import base_result


def is_pdf_scanned(path):
    try:
        doc = fitz.open(path)
        page = doc[0]

        # --- 1) Проверяем наличие текстового слоя ---
        text_layer = page.get_text("text").strip()

        if text_layer:
            # --- 2) Проверяем, не мусор ли текст ---
            letters = sum(c.isalpha() for c in text_layer)
            ratio = letters / max(len(text_layer), 1)

            # Если текст есть, но он мусорный → считаем сканом
            if ratio < 0.2:
                return True

            # Если текст есть и он нормальный → НЕ скан
            return False

        # --- 3) Если текстового слоя нет → скан ---
        return True

    except Exception:
        # Если PDF не читается → считаем сканом
        return True

def ocr_backend_for_type(file_type: str) -> str:
    if file_type == "pdf_scans":
        return "paddle"

    if file_type == "images":
        return "paddle"

    return "extract_text"

def detect_file_type(path: Path):
    # Если это PDF — проверяем наличие текстового слоя на первых страницах
    if path.suffix.lower() == ".pdf":
        try:
            doc = fitz.open(path)
            text_len = 0

            # Проверяем первые 3 страницы (если есть)
            for i in range(min(3, len(doc))):
                page_text = doc[i].get_text().strip()
                text_len += len(page_text)

            # Если есть текстовый слой → pdf_text
            if text_len > 50:
                return "pdf_text"
            else:
                return "pdf_scans"

        except Exception:
            return "pdf_scans"

    # Остальные типы — как раньше
    parent = path.parent.name
    if parent in ["pdf_text", "pdf_scans", "doc_Word", "tables", "images", "txt"]:
        return parent

    return "unknown"

# ---------------------------------------------------------
# PAGE COUNT
# ---------------------------------------------------------
def get_pdf_pages(path):
    try:
        import fitz
        doc = fitz.open(path)
        return len(doc)
    except:
        return None

def get_word_pages(path):
    path = Path(path)
    temp_docx = None

    # Если .doc → конвертируем
    if path.suffix.lower() == ".doc":
        temp_docx = convert_doc_to_docx(path)
        path = temp_docx

    # Конвертация DOCX → PDF
    temp_pdf = convert_docx_to_pdf(path)
    if temp_pdf is None:
        return None

    # Подсчёт страниц
    try:
        with pdfplumber.open(temp_pdf) as pdf:
            pages = len(pdf.pages)
    except:
        pages = None

    # Удаляем PDF
    try:
        temp_pdf.unlink()
    except:
        pass

    # Удаляем временный DOCX
    if temp_docx:
        try:
            temp_docx.unlink()
        except:
            pass

    return pages

def get_excel_sheets(path):
    try:
        import pandas as pd
        sheets = pd.ExcelFile(path).sheet_names
        return len(sheets)
    except:
        return None

# ---------------------------------------------------------
# LANGUAGE
# ---------------------------------------------------------    
def detect_language_safe_blocks(text):
    ru = sum("а" <= ch <= "я" or "А" <= ch <= "Я" for ch in text)
    en = sum("a" <= ch <= "z" or "A" <= ch <= "Z" for ch in text)
    if ru > en:
        return "ru"
    elif en > ru:
        return "en"
    else:
        return "unknown"

def normalize_for_lang(text):
    return text.replace("\n", " ").replace("\t", " ").strip()

def detect_language_safe(text):
    return detect_language_safe_blocks(text)

# ---------------------------------------------------------
# DETECT TABLES
# ---------------------------------------------------------
def detect_tables_layout(pdf):
    tables = []

    for page_idx, page in enumerate(pdf.pages):
        words = page.extract_words()
        lines = {}

        for w in words:
            y = w["top"]
            found_key = None
            for key in lines.keys():
                if abs(key - y) <= 3:
                    found_key = key
                    break
            if found_key is None:
                lines[y] = []
                found_key = y
            lines[found_key].append(w)

        sorted_lines = []
        for y in sorted(lines.keys()):
            row_words = sorted(lines[y], key=lambda w: w["x0"])
            sorted_lines.append((page_idx + 1, y, row_words))

        rows = []
        for page_num, y, row_words in sorted_lines:
            cols = []
            current_col = [row_words[0]]

            for prev, cur in zip(row_words, row_words[1:]):
                if cur["x0"] - prev["x1"] > 10:
                    cols.append(current_col)
                    current_col = [cur]
                else:
                    current_col.append(cur)
            cols.append(current_col)

            col_texts = [" ".join(w["text"] for w in col) for col in cols]

            rows.append({
                "page": page_num,
                "y": y,
                "cols": col_texts
            })

        grouped = []
        current = []
        last_page = None
        last_y = None
        last_ncols = None

        for row in rows:
            ncols = len(row["cols"])

            if ncols < 3:
                if current:
                    grouped.append(current)
                    current = []
                continue

            if (last_page is None or
                row["page"] != last_page or
                (last_y is not None and abs(row["y"] - last_y) > 20) or
                (last_ncols is not None and ncols != last_ncols)):
                if current:
                    grouped.append(current)
                    current = []
                current.append(row)
            else:
                current.append(row)

            last_page = row["page"]
            last_y = row["y"]
            last_ncols = ncols

        if current:
            grouped.append(current)

        for g in grouped:
            if len(g) >= 3:
                tables.append(g)

    return tables

def detect_tables_from_text(text):
    tables = []
    current = []

    for line in text.split("\n"):
        # Примитивные признаки таблицы
        if "|" in line or "\t" in line or "  " in line:
            current.append([c.strip() for c in line.replace("\t", "|").split("|")])
        else:
            if current:
                tables.append(current)
                current = []
    if current:
        tables.append(current)

    return tables

# ---------------------------------------------------------
# ANALYZE table/image/count
# ---------------------------------------------------------
def analyze_pdf(path: Path):
    tables_struct = []
    images = []

    with pdfplumber.open(path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            try:
                for tbl in page.extract_tables():
                    rows = [[cell or "" for cell in row] for row in tbl]
                    tables_struct.append({"page": page_idx+1, "rows": rows})
            except:
                pass

            for im in page.images:
                images.append({
                    "page": page_idx+1,
                    "bbox": [im["x0"], im["top"], im["x1"], im["bottom"]],
                })

        layout_tables = detect_tables_layout(pdf)

    return {
        "tables_struct": tables_struct,
        "layout_tables": layout_tables,
        "images": images,
        "table_count": len(tables_struct) + len(layout_tables),
        "image_count": len(images),
    }

def analyze_docx(path: Path):
    doc = Document(path)

    tables = []
    for t in doc.tables:
        rows = []
        for row in t.rows:
            rows.append([cell.text.strip() for cell in row.cells])
        tables.append(rows)

    images = []
    for rel in doc.part._rels:
        target = doc.part._rels[rel].target_ref
        if "image" in target:
            images.append(target)

    return {
        "tables": tables,
        "images": images,
        "table_count": len(tables),
        "image_count": len(images),
    }

def analyze_txt(path: Path):
    text = open(path, "r", encoding="utf-8", errors="ignore").read()

    tables = []
    current = []

    for line in text.split("\n"):
        if "|" in line or "\t" in line:
            current.append([c.strip() for c in line.replace("\t", "|").split("|")])
        else:
            if current:
                tables.append(current)
                current = []
    if current:
        tables.append(current)

    return {
        "tables": tables,
        "images": [],
        "table_count": len(tables),
        "image_count": 0,
    }

def analyze_rtf(path: Path):
    raw = open(path, "rb").read()
    text = rtf_to_text(raw.decode("utf-8", errors="ignore"))

    tables = []
    current = []
    for line in text.split("\n"):
        if "|" in line or "\t" in line:
            current.append([c.strip() for c in line.replace("\t", "|").split("|")])
        else:
            if current:
                tables.append(current)
                current = []
    if current:
        tables.append(current)

    images = []
    if b"\\pict" in raw:
        images.append("embedded_image")

    return {
        "tables": tables,
        "images": images,
        "table_count": len(tables),
        "image_count": len(images),
    }

def analyze_excel(path: Path):
    ext = path.suffix.lower()
    tables = []

    if ext == ".xls":
        book = xlrd.open_workbook(str(path))
        for sheet in book.sheets():
            rows = []
            for r in range(sheet.nrows):
                rows.append([str(sheet.cell_value(r, c)) for c in range(sheet.ncols)])
            tables.append({"sheet": sheet.name, "rows": rows})
    else:
        sheets = pd.read_excel(path, sheet_name=None, header=None)
        for name, df in sheets.items():
            rows = []
            for row in df.values:
                rows.append([str(v) for v in row])
            tables.append({"sheet": name, "rows": rows})

    return {
        "tables": tables,
        "images": [],
        "table_count": len(tables),
        "image_count": 0,
    }
# -------------------------
# Онсновная функция extract_text 
# -------------------------
def extract_text(path):
    """
    Универсальный экстрактор.
    Возвращает dict:
    - text
    - source
    - time_sec
    - warnings
    - blocks
    - tables
    - images
    - requires_ocr
    - meta (полная, объединённая)
    """
    ENABLE_EXTRACT_LOGS = False

    file_type = detect_file_type(path)

    step2_dir = Path("results/step2")
    step2_dir.mkdir(parents=True, exist_ok=True)

    extract_log_path = step2_dir / "extract_text_logs.jsonl"
    extract_err_path = step2_dir / "extract_text_errors.jsonl"

    path = Path(path)
    ext = path.suffix.lower()
    start = time.time()
    res = base_result()

    try:
        # -------------------------
        # PDF
        # -------------------------
        if ext == ".pdf":
            native = extract_pdf_text_fitz(path)
            scanned = is_pdf_scanned(path)
            native["requires_ocr"] = scanned

            if scanned:
                ocr_res = extract_pdf_scanned(path)
                if ocr_res["text"].strip():
                    res = native
                    res["text"] = ocr_res["text"]
                    res["source"] = "ocr"
                    res["warnings"].extend(ocr_res["warnings"])
                else:
                    res = native
            else:
                res = native

            # --- OCR-таблицы (если скан)
            ocr_tables = detect_tables_from_text(res["text"]) if scanned else []

            # --- PDF-таблицы (если текстовый PDF)
            pdf_info = analyze_pdf(path)
            pdf_tables = pdf_info["tables_struct"]

            # --- Объединяем
            res["tables"] = ocr_tables + pdf_tables
            res["images"] = pdf_info["images"]

        # -------------------------
        # DOCX / DOC
        # -------------------------
        elif ext in [".docx", ".doc"]:
            res = extract_docx(path)

            if ext == ".doc":
                temp_docx = convert_doc_to_docx(path)
                docx_info = analyze_docx(temp_docx)
                try:
                    temp_docx.unlink()
                except:
                    pass
            else:
                docx_info = analyze_docx(path)

            res["tables"] = docx_info["tables"]
            res["images"] = docx_info["images"]

        # -------------------------
        # XLS / XLSX
        # -------------------------
        elif ext in [".xlsx", ".xls"]:
            res = extract_excel(path)
            excel_info = analyze_excel(path)
            res["images"] = excel_info["images"]

        # -------------------------
        # IMAGES
        # -------------------------
        elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            res = extract_image(path)
            res["requires_ocr"] = True

            # Таблицы из OCR-текста
            res["tables"] = detect_tables_from_text(res["text"])
            res["images"] = [str(path)]

        # -------------------------
        # TXT
        # -------------------------
        elif ext == ".txt":
            res = extract_txt(path)
            txt_info = analyze_txt(path)
            res["tables"] = txt_info["tables"]
            res["images"] = txt_info["images"]

        # -------------------------
        # RTF
        # -------------------------
        elif ext == ".rtf":
            res = extract_rtf(path)
            rtf_info = analyze_rtf(path)
            res["tables"] = rtf_info["tables"]
            res["images"] = rtf_info["images"]

        else:
            res["warnings"].append(f"Unsupported format: {ext}")

    except Exception as e:
        res = base_result()
        res["source"] = "error"
        res["warnings"].append(str(e))
        with open(extract_err_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"file": str(path), "error": str(e)}, ensure_ascii=False) + "\n")

    res["time_sec"] = time.time() - start

    # =========================================================
    # META — ПОЛНАЯ, ОБЪЕДИНЁННАЯ
    # =========================================================

    text = res.get("text", "")

    # PAGES
    if ext == ".pdf":
        pages = get_pdf_pages(path)
    elif ext in [".doc", ".docx"]:
        pages = get_word_pages(path)
    else:
        pages = None

    # SHEETS
    sheets = get_excel_sheets(path) if ext in [".xlsx", ".xls"] else None

    # TABLES / IMAGES / TEXT_LENGTH
    if ext == ".pdf":
        table_count = len(res.get("tables", []))
        image_count = len(res.get("images", []))
        text_length = len(text.strip())

    elif ext in [".doc", ".docx"]:
        table_count = len(res.get("tables", []))
        image_count = len(res.get("images", []))
        text_length = len(text.strip())

    elif ext in [".xls", ".xlsx"]:
        text_length = 0
        for sheet in res.get("sheets", []):
            df_dict = sheet["df"]
            for row in df_dict["data"]:
                text_length += sum(len(str(v)) for v in row)

        table_count = len(res.get("sheets", []))
        image_count = 0

    elif ext == ".txt":
        table_count = len(res.get("tables", []))
        image_count = 0
        text_length = len(text.strip())

    elif ext == ".rtf":
        table_count = len(res.get("tables", []))
        image_count = len(res.get("images", []))
        text_length = len(text.strip())

    elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
        table_count = len(res.get("tables", []))
        image_count = 1
        text_length = len(text.strip())

    else:
        table_count = 0
        image_count = 0
        text_length = len(text.strip())

    # =========================================================
    # LANGUAGE
    # =========================================================
    if ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"] or res.get("requires_ocr", False):
        language = detect_language_safe_blocks(text)
    elif ext in [".xls", ".xlsx"]:
        sample = ""
        for sheet in res.get("sheets", []):
            df_dict = sheet["df"]
            for row in df_dict["data"][:20]:
                sample += " ".join(str(v) for v in row) + " "
        clean = normalize_for_lang(sample)
        language = detect_language_safe(clean[:500]) if len(clean) > 20 else "unknown"
    else:
        clean = normalize_for_lang(text)
        language = detect_language_safe(clean[:500]) if len(clean) > 20 else "unknown"

    # META — финальная
    res["meta"] = {
        "doc_id": path.stem,
        "filename": path.name,
        "file_path": str(path),
        "file_path_v2": str(path.parent),
        "extension": ext,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
        "pages": pages,
        "sheets": sheets,
        "table_count": table_count,
        "image_count": image_count,
        "language": language,
        "requires_ocr": res.get("requires_ocr", False),
        "ocr_backend": res.get("source"),
        "ocr_backend_f": ocr_backend_for_type(file_type),
        "source": detect_file_type(path),
        "time_sec": res.get("time_sec"),
        "text_length": text_length,
        "warnings": res.get("warnings", []),
    }

    # LOG JSONL
    if ENABLE_EXTRACT_LOGS:
        with open(extract_log_path, "a", encoding="utf-8") as f:
            log_entry = res.copy()
            log_entry["file"] = str(path)
            f.write(json.dumps(log_entry, ensure_ascii=False, default=str) + "\n")

    return res

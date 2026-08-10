import re
from src.segmentation.normalization import normalize_text
from src.segmentation.headers import is_header

from src.segmentation.segment_txt import segment_txt
from src.segmentation.segment_docx import segment_docx
from src.segmentation.segment_doc import segment_doc
from src.segmentation.segment_pdf_text import segment_pdf_text
from src.segmentation.segment_pdf_scans import segment_pdf_scans
from src.segmentation.segment_images import segment_images_universal
from src.segmentation.segment_xlsx import segment_xlsx_universal
from src.segmentation.postprocess import segmentation_postprocess

# intermediate_cache будет заполнен в шаге 4
intermediate_cache = {}

def rule_based_segment(text_raw: str, file_type: str | None = None, doc_id: str | None = None):
    if file_type:
        file_type = file_type.strip().lower()

    # ============================
    # 1. Специальные типы файлов
    # ============================

    # XLSX
    if file_type == "tables":
        return segment_xlsx_universal(doc_id)

    # IMAGES
    if file_type == "images":
        return segment_images_universal(doc_id, text_raw)

    # TXT
    if file_type == "txt":
        return segment_txt(text_raw, doc_id)

    # PDF_SCANS (ВАЖНО: принимает только text_raw)
    if file_type == "pdf_scans":
        return segment_pdf_scans(text_raw)

    # PDF_TEXT (ВАЖНО: принимает только text_raw)
    if file_type in ("pdf_text", "pdf_txt"):
        return segment_pdf_text(text_raw)

    # DOCX
    if file_type == "docx":
        return segment_docx(doc_id)

    # DOC (после конвертации → TXT сегментатор)
    if file_type == "doc_word":
        text = normalize_text(text_raw)
        return segment_txt(text, doc_id)

    # ============================
    # 2. Остальные текстовые файлы
    # ============================

    text = normalize_text(text_raw)
    lines = text.splitlines()

    blocks = []
    current_type = None
    current_lines = []

    def flush():
        nonlocal current_type, current_lines
        if current_lines:
            raw = " ".join(current_lines).strip()
            if raw:
                blocks.append({"type": current_type, "text": raw})
        current_type = None
        current_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue

        # FIGURE
        if is_figure_line(stripped):
            flush()
            blocks.append({"type": "figure", "caption": stripped})
            continue

        # HEADER
        if is_header(stripped) and file_type not in ("pdf_text", "pdf_txt"):
            flush()
            blocks.append({"type": "header", "text": stripped})
            continue

        # UNMARKED LIST ITEM
        if is_unmarked_list_item(stripped):
            flush()
            blocks.append({"type": "list", "items": [stripped]})
            continue

        # LIST
        if file_type not in ("pdf_text", "pdf_txt") and is_list_line(stripped):
            # DOC/DOCX short list lines → treat as paragraph
            if file_type in ("doc_word", "doc", "docx") and len(stripped.split()) < 3:
                if current_type not in ("paragraph", None):
                    flush()
                current_type = "paragraph"
                current_lines.append(stripped)
                continue

            if current_type != "list":
                flush()
                current_type = "list"
                current_lines = [stripped]
            else:
                current_lines.append(stripped)
            continue

        # TABLE
        if (
            file_type not in ("pdf_text", "pdf_txt")
            and is_table_line(stripped)
            and file_type not in ("doc", "docx", "doc_word")
        ):
            if current_type != "table":
                flush()
                current_type = "table"
                current_lines = [stripped]
            else:
                current_lines.append(stripped)
            continue

        # PARAGRAPH
        if current_type not in ("paragraph", None):
            flush()
        current_type = "paragraph"
        current_lines.append(stripped)

    flush()
    return segmentation_postprocess(blocks, file_type=file_type)

# -----------------------------
# Разбиение длинных параграфов на чанки
# -----------------------------
def split_long_paragraphs(blocks, file_type=None, max_len=1500):
    new_blocks = []
    for b in blocks:

        # --- не трогаем заголовки и header_paragraph ---
        if b["type"] in ("header", "header_paragraph"):
            new_blocks.append(b)
            continue

        if b["type"] != "paragraph":
            new_blocks.append(b)
            continue

        text = b["text"].strip()
        if len(text) <= max_len:
            new_blocks.append(b)
            continue

        if file_type in ("pdf_scans", "images"):
            new_blocks.append(b)
            continue

        try:
            doc = nlp_en(text)
            sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        except Exception:
            sentences = re.split(r"(?<=[.!?])\s+", text)

        if len(sentences) <= 1:
            new_blocks.append(b)
            continue

        chunk = ""
        for sent in sentences:
            if len(sent) > max_len:
                if chunk:
                    new_blocks.append({"type": "paragraph", "text": chunk.strip()})
                    chunk = ""
                for i in range(0, len(sent), max_len):
                    new_blocks.append({
                        "type": "paragraph",
                        "text": sent[i:i+max_len].strip()
                    })
                continue

            if len(chunk) + len(sent) < max_len:
                chunk += " " + sent
            else:
                if chunk.strip():
                    new_blocks.append({"type": "paragraph", "text": chunk.strip()})
                chunk = sent

        if chunk.strip():
            new_blocks.append({"type": "paragraph", "text": chunk.strip()})

    return new_blocks

def get_pred_text(block):
    t = block["type"]

    # --- TEXT BLOCKS ---
    if t == "paragraph":
        return block.get("text", "")

    if t == "header":
        return block.get("text", "")

    if t == "header_paragraph":
        return block.get("text", "")

    if t == "list":
        return "\n".join(block.get("items", []))

    if t == "figure":
        return block.get("caption", "")

    # --- TABLE BLOCKS ---
    if t == "table":
        cols = block.get("columns")
        data = block.get("data")        # XLSX tables
        rows = block.get("rows")        # PDF/TXT tables
        content = block.get("content")  # PDF tables

        out = []

        if cols:
            out.append(" | ".join(map(str, cols)))

        if data:
            for r in data[:50]:
                out.append(" | ".join(map(str, r)))

        if rows:
            for r in rows[:50]:
                out.append(" | ".join(map(str, r)))

        if content:
            for r in content[:50]:
                out.append(" | ".join(map(str, r)))

        return "\n".join(out)

    return ""

from src.segmentation.headers import is_header
from src.segmentation.postprocess import segmentation_postprocess

# -----------------------------
# Сегментатор PDF_SCANS (заголовки + параграфы)
# -----------------------------
def segment_pdf_scans(text_raw: str) -> list[dict]:
    """
    Упрощённый сегментатор для PDF_SCANS:
    - заголовки
    - параграфы
    - без таблиц и списков (слишком много FP)
    """
    text = text_raw
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

        # HEADER
        if is_header(stripped):
            flush()
            blocks.append({"type": "header", "text": stripped})
            continue

        # PARAGRAPH
        if current_type not in ("paragraph", None):
            flush()
        current_type = "paragraph"
        current_lines.append(stripped)

    flush()
    return segmentation_postprocess(blocks, file_type="pdf_scans")
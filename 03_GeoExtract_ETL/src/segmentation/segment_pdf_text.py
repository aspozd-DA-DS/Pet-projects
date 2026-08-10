import re
from src.segmentation.normalization import normalize_text_pdf
from src.segmentation.headers import is_header

from src.segmentation.postprocess import segmentation_postprocess

def segment_pdf_text(text_raw: str) -> list[dict]:
    """
    Улучшенный сегментатор для PDF_TEXT (версия 3.0):
    - корректная нормализация без разрушения структуры
    - восстановление предложений
    - улучшенная логика заголовков (включая 3.1, 3.2.1)
    - улучшенная логика списков (включая bullets, unicode)
    - улучшенная логика таблиц (включая whitespace-aligned)
    - корректная сегментация параграфов
    """

    # 1. Мягкая нормализация
    text = normalize_text_pdf(text_raw)
    lines = text.splitlines()

    # 2. Удаляем мусорные строки
    clean = []
    for ln in lines:
        s = ln.strip()
        if not s:
            clean.append("")
            continue
        if len(s) < 3 and not s.isalpha():
            continue
        clean.append(s)

    # 3. Восстановление предложений
    merged = []
    buf = ""

    def is_sentence_end(s):
        return s.endswith((".", "?", "!", ";", ":"))

    for ln in clean:
        if not ln:
            if buf:
                merged.append(buf.strip())
                buf = ""
            merged.append("")
            continue

        if buf and ln[0].islower():
            buf += " " + ln
            continue

        if buf and not is_sentence_end(buf):
            buf += " " + ln
            continue

        if buf:
            merged.append(buf.strip())
        buf = ln

    if buf:
        merged.append(buf.strip())

    # 4. Сегментация
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

    # --- эвристики ---
    def is_header_line(s):
        if re.match(r"^\d+(\.\d+){1,3}\s+[A-Z]", s):
            return True
        if re.match(r"^\d+[\.\)]\s+[A-Z]", s):
            return True
        if s.istitle() and len(s.split()) <= 10:
            return True
        if is_header(s):
            return True
        return False

    def is_list_line(s):
        if s.startswith(("•", "-", "–", "—", "*", "▪", "»", "›", "→")):
            return True
        if re.match(r"^\(?\d+[\.\)]\s+", s):
            return True
        if re.match(r"^\(?[a-zA-Z][\.\)]\s+", s):
            return True
        return False

    def is_table_line(s):
        tokens = s.split()
        if len(tokens) >= 3:
            nums = sum(1 for t in tokens if re.match(r"^[\d\.,\-]+$", t))
            if nums >= 2:
                return True
        if re.search(r"\s{4,}", s):
            return True
        return False

    # --- основной цикл ---
    for ln in merged:
        stripped = ln.strip()

        if not stripped:
            flush()
            continue

        if is_header_line(stripped):
            flush()
            blocks.append({"type": "header", "text": stripped})
            continue

        if is_list_line(stripped):
            flush()
            blocks.append({"type": "list", "items": [stripped]})
            continue

        if is_table_line(stripped):
            flush()
            blocks.append({"type": "table", "rows": [stripped.split()]})
            continue

        if current_type not in ("paragraph", None):
            flush()
        current_type = "paragraph"
        current_lines.append(stripped)

    flush()

    return segmentation_postprocess(blocks, file_type="pdf_text")
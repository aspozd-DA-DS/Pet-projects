import re

# -----------------------------
# Мягкая нормализация для DOC/DOCX/TXT
# -----------------------------
def normalize_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        s = line.strip()
        if not s:
            cleaned.append("")
            continue
        if len(s) <= 2 and not s.isalpha():
            continue
        cleaned.append(s)

    merged = []
    buf = ""
    
    # Эвристика конца предложения (., ?, !, ;)
    def is_sentence_end(s):
        return s.endswith((".", "?", "!", ":", ";"))

    for line in cleaned:
        if not line:
            if buf:
                merged.append(buf.strip())
                buf = ""
            merged.append("")
            continue
        if buf and not is_sentence_end(buf):
            buf += " " + line
        else:
            if buf:
                merged.append(buf.strip())
            buf = line

    if buf:
        merged.append(buf.strip())

    out = []
    prev_empty = False
    for line in merged:
        if not line.strip():
            if not prev_empty:
                out.append("")
            prev_empty = True
        else:
            out.append(line)
            prev_empty = False

    return "\n".join(out)

# -----------------------------
# Нормализация для PDF_TEXT: склейка переносов, без агрессивного объединения абзацев
# -----------------------------
def normalize_text_pdf(text_raw: str) -> str:
    """
    Мягкая нормализация PDF-текста:
    - не ломает структуру
    - убирает мусор
    - нормализует пробелы
    - восстанавливает переносы
    """

    lines = text_raw.splitlines()
    out = []

    for ln in lines:
        s = ln.rstrip()

        # удаляем мусорные строки
        if len(s.strip()) == 0:
            out.append("")
            continue

        # удаляем строки из 1–2 символов, если это не буквы
        if len(s.strip()) < 3 and not s.strip().isalpha():
            continue

        # нормализация пробелов
        s = re.sub(r"\s+", " ", s).strip()

        out.append(s)

    return "\n".join(out)
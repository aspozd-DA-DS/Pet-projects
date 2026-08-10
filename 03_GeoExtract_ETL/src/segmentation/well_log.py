import re

# -----------------------------
# WELL-LOG
# -----------------------------
def is_well_log(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 5:
        return False

    header_idx = None
    for i, ln in enumerate(lines[:50]):
        up = ln.upper()
        if "DEPTH" in up:
            tokens = ln.split()
            if len(tokens) >= 6:
                header_idx = i
                break

    if header_idx is None:
        return False

    numeric_lines = 0
    for ln in lines[header_idx+1 : header_idx+200]:
        tokens = ln.split()
        nums = sum(1 for t in tokens if re.match(r"^[\d\.\-]+$", t))
        if nums >= 3:
            numeric_lines += 1

    return numeric_lines >= 3

# -----------------------------
# Сегментатор WELL‑LOG (описание + таблица)
# -----------------------------
def segment_well_log(text_raw: str):
    lines = [ln.strip() for ln in text_raw.splitlines()]

    # 1. Описание кривых (до пустой строки)
    header_lines = []
    i = 0
    while i < len(lines):
        if lines[i] == "":
            break
        header_lines.append(lines[i])
        i += 1

    paragraph_text = "\n".join(header_lines)

    # 2. Найти строку с названиями колонок
    columns = None
    for j in range(i+1, len(lines)):
        if lines[j].startswith("DEPTH"):
            columns = lines[j].split()
            i = j + 1
            break

    # 3. Собрать строки таблицы
    rows = []
    for ln in lines[i:]:
        if not ln.strip():
            continue
        tokens = ln.split()
        # строки таблицы имеют ≥3 числовых значений
        nums = sum(1 for t in tokens if re.match(r"^[\d\.\-]+$", t))
        if nums >= 3:
            rows.append(tokens)

    return [
        {
            "type": "paragraph",
            "start_page": 1,
            "text": paragraph_text
        },
        {
            "type": "table",
            "start_page": 1,
            "columns": columns,
            "rows": rows
        }
    ]
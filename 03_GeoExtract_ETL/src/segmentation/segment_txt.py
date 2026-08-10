import re

# -----------------------------
# TXT — простой сегментатор
# -----------------------------
def segment_txt(text_raw: str, doc_id: str):
    lines = [l.strip() for l in text_raw.splitlines()]
    lines = [l for l in lines if l]

    blocks = []
    buffer = ""

    def flush():
        nonlocal buffer
        if buffer.strip():
            blocks.append({"type": "paragraph", "text": buffer.strip()})
        buffer = ""

    for line in lines:

        # FIGURE
        if re.match(r"^(Figure|Fig\.)\s*\d+", line, re.IGNORECASE):
            flush()
            blocks.append({"type": "figure", "caption": line})
            continue

        # HEADER (Golden Set style)
        if (
            len(line) < 50
            and line[0].isupper()
            and line.replace(" ", "").isalpha()
        ):
            flush()
            blocks.append({"type": "header", "text": line})
            continue

        # LIST (Golden Set style)
        if (
            re.match(r"^(\d+[\.\)]|[-*•])\s+", line)
            or (len(line) < 50 and line.count(" ") <= 3)
        ):
            flush()
            blocks.append({"type": "list", "items": [line]})
            continue

        # Aggressive merging
        if len(line) < 100:
            buffer += " " + line
            continue

        flush()
        blocks.append({"type": "paragraph", "text": line})

    flush()
    return blocks
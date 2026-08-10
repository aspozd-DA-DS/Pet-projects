import json
import re

# -----------------------------
# LOAD OCR FIXES
# -----------------------------
def load_ocr_fixes(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# normalize_block_text
# -----------------------------
def normalize_block_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\u200b", " ").replace("\xa0", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\x00-\x1F]", " ", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"(\d+),(\d+)", r"\1.\2", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = re.sub(r"([.,;:])\s+", r"\1 ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# -----------------------------
# fix_ocr_terms
# -----------------------------
def fix_ocr_terms(text: str, ocr_fixes: dict) -> str:
    if not text:
        return text

    for wrong, correct in ocr_fixes.items():
        pattern = r"\b" + re.escape(wrong) + r"\b"
        text = re.sub(pattern, correct, text, flags=re.IGNORECASE)

    return text

# -----------------------------
# normalize_blocks
# -----------------------------
def normalize_blocks(blocks: list, ocr_fixes: dict) -> list:
    normalized = []

    for block in blocks:
        b = block.copy()

        if b["type"] in ("paragraph", "header", "list", "figure"):
            if "text" in b:
                b["text"] = fix_ocr_terms(normalize_block_text(b["text"]), ocr_fixes)

            if b["type"] == "figure" and "caption" in b:
                b["caption"] = fix_ocr_terms(normalize_block_text(b["caption"]), ocr_fixes)

            if b["type"] == "list" and "items" in b:
                b["items"] = [
                    fix_ocr_terms(normalize_block_text(it), ocr_fixes)
                    for it in b["items"]
                ]

        elif b["type"] == "table":
            if "caption" in b:
                b["caption"] = fix_ocr_terms(normalize_block_text(b["caption"]), ocr_fixes)

            if "columns" in b:
                b["columns"] = [normalize_block_text(c) for c in b["columns"]]

            if "rows" in b:
                b["rows"] = [
                    [normalize_block_text(str(cell)) for cell in row]
                    for row in b["rows"]
                ]

            if "data" in b:
                b["data"] = [
                    [normalize_block_text(str(cell)) for cell in row]
                    for row in b["data"]
                ]

            if "header_descriptions" in b:
                b["header_descriptions"] = {
                    k: normalize_block_text(v)
                    for k, v in b["header_descriptions"].items()
                }

        normalized.append(b)

    return normalized

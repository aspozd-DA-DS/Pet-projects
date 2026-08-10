import json
import re

# -----------------------------
# LOAD REGEX PATTERNS
# -----------------------------
def load_patterns_regex(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# NORMALIZE NUMERIC
# -----------------------------
def normalize_numeric(v):
    try:
        return float(v)
    except:
        return v

# -----------------------------
# EXTRACT FEATURES REGEX FROM BLOCK
# -----------------------------
def extract_features_regex_from_block(block, patterns_regex):
    result = {k: [] for k in patterns_regex}

    def check(text):
        if not text:
            return
        for key, pattern in patterns_regex.items():
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            for m in matches:
                result[key].append(normalize_numeric(m))

    t = block["type"]

    if t in ("header", "paragraph"):
        check(block.get("text", ""))

    elif t == "figure":
        check(block.get("caption", ""))

    elif t == "list":
        for item in block.get("items", []):
            check(item)

    elif t == "table":
        table = block.get("table", {}) or {}
        check(table.get("caption", ""))

        for col in table.get("columns") or []:
            check(str(col))

        for row in table.get("data") or []:
            for cell in row:
                check(str(cell))

        for row in table.get("rows") or []:
            for cell in row:
                check(str(cell))

        for k, v in (table.get("header_descriptions") or {}).items():
            check(k)
            check(v)

        # OLD FORMAT SUPPORT
        check(block.get("caption", ""))
        for col in block.get("columns") or []:
            check(str(col))
        for row in block.get("rows") or []:
            for cell in row:
                check(str(cell))
        for k, v in (block.get("header_descriptions") or {}).items():
            check(k)
            check(v)
        check(block.get("text", ""))

    return result

# -----------------------------
# EXTRACT FEATURES REGEX (DOC LEVEL)
# -----------------------------
def extract_features_regex(blocks, patterns_regex):
    final = {k: [] for k in patterns_regex}

    for block in blocks:
        feats = extract_features_regex_from_block(block, patterns_regex)
        for k, vals in feats.items():
            final[k].extend(vals)

    for k in final:
        cleaned = [v for v in final[k] if v not in ("", None)]
        if not cleaned:
            final[k] = []
            continue

        all_float = all(isinstance(v, float) for v in cleaned)
        all_str = all(isinstance(v, str) for v in cleaned)

        if all_float or all_str:
            final[k] = sorted(set(cleaned))
        else:
            final[k] = list(set(cleaned))

    return final

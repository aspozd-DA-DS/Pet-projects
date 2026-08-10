from striprtf.striprtf import rtf_to_text
from pathlib import Path

from .base import wrap_extractor, base_result

@wrap_extractor
def extract_rtf(path: Path):
    res = base_result()   # ← ИСПРАВЛЕНО
    res["source"] = "native_rtf"

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            res["text"] = rtf_to_text(f.read())
        return res
    except Exception as e:
        res["source"] = "error"
        res["warnings"].append(str(e))
        return res

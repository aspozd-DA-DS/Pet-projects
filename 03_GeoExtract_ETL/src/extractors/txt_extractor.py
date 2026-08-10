import chardet
from pathlib import Path

from .base import wrap_extractor, base_result

@wrap_extractor
def extract_txt(path: Path):
    res = base_result()   # ← ИСПРАВЛЕНО
    res["source"] = "native_txt"

    try:
        raw = open(path, "rb").read()
        enc = chardet.detect(raw)["encoding"]
        res["text"] = raw.decode(enc or "utf-8", errors="ignore")
        return res
    except Exception as e:
        res["source"] = "error"
        res["warnings"].append(str(e))
        return res

import json
import csv

# -----------------------------
# group_blocks_by_doc
# -----------------------------
def group_blocks_by_doc(path: str) -> dict:
    docs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.setdefault(obj["doc_id"], []).append(obj)
    return docs

# -----------------------------
# load_step2_full
# -----------------------------
def load_step2_full(path: str) -> dict:
    """
    Загружает полный результат extract_text():
    - meta
    - text
    - tables
    - images
    """
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_id = obj["meta"]["doc_id"]

            data[doc_id] = {
                "meta": obj.get("meta", {}),
                "text": obj.get("text", ""),
                "tables": obj.get("tables", []),
                "images": obj.get("images", [])
            }

    return data

# -----------------------------
# build_document_json
# -----------------------------
def build_document_json(doc_id: str, blocks: list, meta: dict) -> dict:
    return {
        "doc_id": doc_id,
        "document_type": meta.get("document_type"),
        "extension": meta.get("extension"),
        "language": meta.get("language"),
        "requires_ocr": meta.get("requires_ocr"),
        "total_blocks": len(blocks),
        "blocks": blocks,
        "source_text_quality": {
            "ocr_used": meta.get("requires_ocr"),
            "ocr_confidence": meta.get("ocr_confidence"),
            "num_blocks": len(blocks)
        }
    }

# -----------------------------
# clean_empty_fields
# -----------------------------
def clean_empty_fields(d: dict, keys: list):
    for k in keys:
        if k in d and (not d[k] or len(d[k]) == 0):
            del d[k]

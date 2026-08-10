# -----------------------------
# IMAGES — универсальный сегментатор
# -----------------------------
def segment_images_universal(doc_id: str, text_raw: str):
    # OCR-текст уже есть в text_raw
    if not text_raw.strip():
        return []
    return [{"type": "paragraph", "text": text_raw.strip()}]
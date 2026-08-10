# -----------------------------
# Пост‑обработка сегментов (склейка коротких параграфов)
# -----------------------------
def segmentation_postprocess(blocks, file_type=None):
    # TXT не трогаем вообще
    if file_type == "txt":
        out = []
        for b in blocks:
            # Преобразуем unmarked list в формат Golden Set
            if b["type"] == "list" and "items" not in b:
                items = b["text"].split() if "text" in b else []
                out.append({
                    "type": "list",
                    "items": items,
                    "start_page": 1
                })
            else:
                out.append(b)
        return out

    out = []
    for b in blocks:
        if (
            out
            and b["type"] == "paragraph"
            and out[-1]["type"] == "paragraph"
            and len(b["text"]) < 40
        ):
            out[-1]["text"] += " " + b["text"]
            continue

        if (
            out
            and b["type"] == "header"
            and out[-1]["type"] == "paragraph"
            and len(b["text"].split()) <= 2
        ):
            out[-1]["text"] += " " + b["text"]
            continue

        out.append(b)

    return out
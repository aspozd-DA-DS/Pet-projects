import json
import numpy as np

def embed_chunks(chunks, model):
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, batch_size=32, show_progress_bar=False)
    return np.asarray(emb, dtype="float32")


def save_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def save_mapping(chunks, path):
    mapping = {
        str(i): {
            "doc_id": c["doc_id"],
            "block_id": c["block_id"]
        }
        for i, c in enumerate(chunks)
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    return mapping

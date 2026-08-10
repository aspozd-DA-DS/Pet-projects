from collections import defaultdict
from src.semantic_search.loaders import load_jsonl, load_doc_types
from src.semantic_search.prepare import prepare_chunk_text, get_features


def build_chunks(path_normalized, path_structured, path_doc_types):
    blocks = load_jsonl(path_normalized)
    doc_types = load_doc_types(path_doc_types)
    structured = {d["doc_id"]: d for d in load_jsonl(path_structured)}

    docs_full = []
    docs_faiss = []

    grouped = defaultdict(list)
    for block in blocks:
        txt = prepare_chunk_text(block)
        if txt and len(txt) > 20:
            grouped[block["doc_id"]].append((block, txt))

    for doc_id, items in grouped.items():
        full_text = "\n".join(txt for _, txt in items)
        if len(full_text) > 5000:
            full_text = full_text[:5000]

        docs_full.append({
            "doc_id": doc_id,
            "block_id": doc_id + "_merged",
            "type": "merged",
            "doc_type": doc_types.get(doc_id, "unknown"),
            "features": get_features(doc_id, structured),
            "text": full_text
        })

        docs_faiss.append({
            "doc_id": doc_id,
            "block_id": doc_id + "_merged",
            "text": full_text
        })

    return docs_full, docs_faiss

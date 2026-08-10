import numpy as np
import faiss

def normalize_query(q):
    return q.strip().lower()


def search(query, model, index, chunks, top_k=3):
    q = normalize_query(query)
    q_emb = model.encode([q])
    q_emb = np.asarray(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)

    scores, idxs = index.search(q_emb, top_k)
    scores, idxs = scores[0], idxs[0]

    results = []
    for score, idx in zip(scores, idxs):
        if idx < 0:
            continue
        c = chunks[idx]
        results.append({
            "doc_id": c["doc_id"],
            "block_id": c["block_id"],
            "score": float(score),
            "text": c["text"]
        })

    return {"query": query, "results": results}

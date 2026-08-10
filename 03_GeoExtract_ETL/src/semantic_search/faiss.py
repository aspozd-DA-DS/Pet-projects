import faiss

def build_faiss_index(emb):
    if emb is None or len(emb) == 0:
        raise ValueError("Empty embeddings")

    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index

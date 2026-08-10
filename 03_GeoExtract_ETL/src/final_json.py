import json

def build_final_json(
    doc_step6,
    step2_data,
    chunks_faiss,
    qs_results,
    cluster_keywords,
    suggestions
):
    doc_id = doc_step6["doc_id"]

    # semantic_hits: выбираем только результаты для данного документа
    hits = []
    for q in qs_results:
        for r in q["results"]:
            if r["doc_id"] == doc_id:
                hits.append({
                    "query": q["query"],
                    "score": r["score"],
                    "block_id": r["block_id"]
                })

    # semantic_chunks: только чанки данного документа
    chunks_for_doc = [c for c in chunks_faiss if c["doc_id"] == doc_id]

    return {
        "doc_id": doc_id,

        # extract_text data (полный набор)
        "meta": step2_data[doc_id]["meta"],
        "raw_text": step2_data[doc_id]["text"],
        "tables": step2_data[doc_id]["tables"],
        "images": step2_data[doc_id]["images"],

        # классификация
        "doc_type": doc_step6.get("doc_type"),
        "classification_confidence": doc_step6.get("classification_confidence"),
        "model_predicted_label": doc_step6.get("model_predicted_label"),
        "model_confidence": doc_step6.get("model_confidence"),

        # блоки документа
        "blocks": doc_step6.get("blocks", []),
        "extracted_features_regex": doc_step6.get("extracted_features_regex", {}),

        # семантика
        "semantic_chunks": chunks_for_doc,
        "semantic_hits": hits,

        # keywords + suggestions
        "keywords": cluster_keywords,
        "suggestions": suggestions
    }


def save_final_json(docs, path):
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

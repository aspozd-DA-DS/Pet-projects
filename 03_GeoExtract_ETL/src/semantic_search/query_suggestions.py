import json
from src.semantic_search.search import search


def load_test_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_query_suggestions(test_queries, model, index, chunks, top_k=3):
    results = []

    for q in test_queries:
        out = search(q, model, index, chunks, top_k=top_k)
        results.append({
            "query": q,
            "results": out["results"]
        })

    return results


def save_query_suggestions(results, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

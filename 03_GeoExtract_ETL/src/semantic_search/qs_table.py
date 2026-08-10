import json
from collections import defaultdict


def build_qs_table(qs_results):
    stats = defaultdict(lambda: {"hits": 0, "scores": []})

    for item in qs_results:
        for r in item["results"]:
            doc_id = r["doc_id"]
            score = r["score"]

            stats[doc_id]["hits"] += 1
            stats[doc_id]["scores"].append(score)

    table = []
    for doc_id, info in stats.items():
        scores = info["scores"]
        table.append({
            "doc_id": doc_id,
            "hits": info["hits"],
            "score_min": min(scores),
            "score_max": max(scores)
        })

    return table


def save_qs_table(table, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)

import json

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def load_doc_types(path):
    d = {}
    for obj in load_jsonl(path):
        d[obj["doc_id"]] = obj.get("doc_type", "unknown")
    return d

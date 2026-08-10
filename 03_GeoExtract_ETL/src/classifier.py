import joblib
import numpy as np
import re

CONFIDENCE_THRESHOLD = 0.55

# ============================================
# LOAD MODELS
# ============================================
def load_tfidf(tfidf_path):
    return joblib.load(tfidf_path)


def load_classifier(classifier_path):
    return joblib.load(classifier_path)

# ============================================
# TEXT EXTRACTION (IDENTICAL TO TRAINING)
# ============================================
def get_document_text(blocks):
    parts = []

    # 1) header + paragraph
    for b in blocks:
        if b.get("type") in ("header", "paragraph", "header_paragraph"):
            txt = b.get("text", "")
            if txt:
                parts.append(txt)

    text = "\n".join(parts).strip()

    # 2) add list items if text is short
    if len(text) < 300:
        list_parts = []
        for b in blocks:
            if b.get("type") == "list":
                items = b.get("items") or []
                for it in items:
                    if isinstance(it, str):
                        list_parts.append(it)
        if list_parts:
            text = text + "\n" + "\n".join(list_parts)

    # 3) fallback: tables
    if len(text.strip()) == 0:
        table_parts = []

        for b in blocks:
            if b.get("type") == "table":

                cap = b.get("caption")
                if cap:
                    table_parts.append(cap)

                cols = b.get("columns") or []
                for col in cols:
                    col_text = re.sub(r"\b\d+\b", "", str(col))
                    col_text = re.sub(r"\s+", " ", col_text).strip()
                    if col_text:
                        table_parts.append(col_text)

                rows = b.get("rows") or []
                for row in rows:
                    row_text = " ".join(str(x) for x in row)
                    row_text = re.sub(r"\b\d+\b", "", row_text)
                    row_text = re.sub(r"\s+", " ", row_text).strip()
                    if row_text:
                        table_parts.append(row_text)

                data = b.get("data")
                if isinstance(data, list):
                    for row in data:
                        row_text = " ".join(str(x) for x in row)
                        row_text = re.sub(r"\b\d+\b", "", row_text)
                        row_text = re.sub(r"\s+", " ", row_text).strip()
                        if row_text:
                            table_parts.append(row_text)

                content = b.get("content") or []
                if isinstance(content, list):
                    for row in content:
                        row_text = " ".join(str(x) for x in row)
                        row_text = re.sub(r"\b\d+\b", "", row_text)
                        row_text = re.sub(r"\s+", " ", row_text).strip()
                        if row_text:
                            table_parts.append(row_text)

                hdr = b.get("header_descriptions") or []
                for h in hdr:
                    h_text = re.sub(r"\b\d+\b", "", str(h))
                    h_text = re.sub(r"\s+", " ", h_text).strip()
                    if h_text:
                        table_parts.append(h_text)

                if "text" in b:
                    t = b.get("text")
                    if isinstance(t, str):
                        t_clean = re.sub(r"\b\d+\b", "", t)
                        t_clean = re.sub(r"\s+", " ", t_clean).strip()
                        if t_clean:
                            table_parts.append(t_clean)

        if table_parts:
            text = "\n".join(table_parts).strip()

    return text

# ============================================
# CONFIDENCE (predict_proba / softmax)
# ============================================
def get_model_confidence(model, X_vec):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_vec)[0]
        return float(np.max(probs))

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_vec)[0]
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        return float(np.max(probs))

    return None

# ============================================
# CLASSIFY DOCUMENT (IDENTICAL TO TRAINING)
# ============================================
def classify_document(doc_json, tfidf, model):
    blocks = doc_json.get("blocks", [])
    text = get_document_text(blocks)

    if not text.strip():
        return {
            "doc_type": "unknown",
            "confidence": 0.0,
            "routing_warning": "empty_text",
            "model_predicted_label": None,
            "model_confidence": 0.0
        }

    X_vec = tfidf.transform([text])
    pred_label = model.predict(X_vec)[0]

    max_prob = get_model_confidence(model, X_vec)
    confidence = max_prob if max_prob is not None else 0.0

    if confidence >= CONFIDENCE_THRESHOLD:
        return {
            "doc_type": pred_label,
            "confidence": confidence,
            "routing_warning": False,
            "model_predicted_label": pred_label,
            "model_confidence": confidence
        }

    return {
        "doc_type": "unknown",
        "confidence": confidence,
        "routing_warning": "low_confidence_classification",
        "model_predicted_label": pred_label,
        "model_confidence": confidence
    }

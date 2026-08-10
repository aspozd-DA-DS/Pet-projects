import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def extract_keywords(chunks, n_clusters=10, top_k=3):
    texts = [c["text"] for c in chunks]

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    cluster_keywords = {}

    for cluster_id in range(n_clusters):
        idxs = (labels == cluster_id).nonzero()[0]
        if len(idxs) == 0:
            continue

        # средний TF-IDF по документам кластера
        cluster_tfidf = X[idxs].mean(axis=0).A1
        top_indices = cluster_tfidf.argsort()[::-1][:top_k]
        keywords = [terms[i] for i in top_indices]

        cluster_keywords[cluster_id] = keywords

    return cluster_keywords


def generate_suggestions(cluster_keywords):
    suggestions = {}

    for cluster_id, keywords in cluster_keywords.items():
        sug = [f"{kw} analysis" for kw in keywords]
        suggestions[cluster_id] = sug

    return suggestions


def save_suggestions(cluster_keywords, suggestions, path):
    data = {
        "keywords": cluster_keywords,
        "suggestions": suggestions
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

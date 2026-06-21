"""Refit each chosen granularity deterministically and persist the FULL state we
might need later — without re-calling the LLM (labels are reused from topics.csv).

Stored (no recompute of labels):
  output/embeddings.npy                     shared doc embeddings (786 x 1024)
  output/mts_<k>/model/                      BERTopic model (safetensors)
  output/mts_<k>/topic_distr.npy             full (786 x n_topics) distribution matrix

Safety: each refit's assignments are compared against the saved article_topics.csv.
Labels are attached by topic_id, which is only valid if assignments match exactly —
so a mismatch aborts that size instead of storing wrong labels.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

# --- config mirrored from the notebook -------------------------------------
BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "revolut_help_articles.jsonl"
OUTPUT_DIR = BASE / "output"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda"
NGRAM_RANGE = (1, 2)
MIN_DF = 2
TOP_N_WORDS = 10
RANDOM_STATE = 42
CHOSEN = [3, 5, 8, 15]


def build_model(min_topic_size):
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric="cosine", random_state=RANDOM_STATE)
    hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size, metric="euclidean",
                            cluster_selection_method="eom", prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=NGRAM_RANGE, min_df=MIN_DF)
    return BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer_model, top_n_words=TOP_N_WORDS,
                    calculate_probabilities=False, verbose=False)


def load_docs():
    articles = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    titles = [a["title"] for a in articles]
    docs = [f"{a['title']}\n\n{a['content_text']}" for a in articles]
    return titles, docs


def get_embeddings(docs):
    path = OUTPUT_DIR / "embeddings.npy"
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    emb = np.asarray(emb, dtype=np.float32)
    np.save(path, emb)
    print(f"embeddings: {emb.shape} -> {path}")
    return emb


def persist_size(mts, docs, titles, embeddings):
    outdir = OUTPUT_DIR / f"mts_{mts}"
    saved = pd.read_csv(outdir / "article_topics.csv")          # ground truth assignments + labels
    saved_topics = saved["topic_id"].tolist()

    model = build_model(mts)
    topics, _ = model.fit_transform(docs, embeddings)
    before = sum(1 for t in topics if t == -1)
    if before > 0:
        topics = model.reduce_outliers(docs, topics, strategy="c-tf-idf")
    model.update_topics(docs, topics=topics,
                        vectorizer_model=CountVectorizer(stop_words="english",
                                                         ngram_range=NGRAM_RANGE, min_df=MIN_DF))

    # --- safety gate: assignments must match the saved run exactly -----------
    matches = sum(int(a == b) for a, b in zip(topics, saved_topics))
    pct = matches / len(topics)
    status = "OK" if matches == len(topics) else "MISMATCH"
    print(f"  mts={mts}: assignment match {matches}/{len(topics)} ({pct:.1%}) [{status}]")
    if matches != len(topics):
        print(f"  mts={mts}: ABORTED — refit diverged from saved run; not storing "
              f"(labels would be misattached). Likely a library/version drift.")
        return {"mts": mts, "ok": False, "match_pct": pct}

    # --- reuse saved LLM labels (no API calls) -------------------------------
    label_csv = pd.read_csv(outdir / "topics.csv")
    label_by_topic = dict(zip(label_csv["topic_id"].astype(int), label_csv["llm_label"]))
    model.set_topic_labels({int(k): str(v) for k, v in label_by_topic.items()})

    # --- full distribution matrix for flexible ambiguity later ---------------
    topic_distr, _ = model.approximate_distribution(docs, window=8, stride=4)
    np.save(outdir / "topic_distr.npy", np.asarray(topic_distr, dtype=np.float32))

    # --- persist the model ---------------------------------------------------
    model_dir = outdir / "model"
    model.save(str(model_dir), serialization="safetensors", save_embedding_model=False)

    n_topics = len([t for t in set(topics) if t != -1])
    print(f"  mts={mts}: stored model/ + topic_distr.npy {topic_distr.shape}, {n_topics} topics")
    return {"mts": mts, "ok": True, "match_pct": pct, "n_topics": n_topics}


def main():
    titles, docs = load_docs()
    print(f"docs: {len(docs)}")
    embeddings = get_embeddings(docs)
    results = [persist_size(mts, docs, titles, embeddings) for mts in CHOSEN]
    print("\nSummary:")
    for r in results:
        print(f"  mts_{r['mts']}: {'stored' if r['ok'] else 'ABORTED'} (match {r['match_pct']:.1%})")


if __name__ == "__main__":
    main()

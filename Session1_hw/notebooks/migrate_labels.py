"""Migration: rebuild LLM-labeled artifacts from saved per-size outputs.

Disk-only — no model, no embeddings, no recompute. Produces:
  output/topics_all_sizes.csv               combined topic table across every mts_<k>
  output/mts_<k>/barchart_labeled.html      barchart titled by "llm_label (count)", all topics
  output/mts_<k>/documents_labeled.html     doc scatter colored by llm_label
                                            (2D coords reused from the original documents.html)
"""
import base64
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
TOP_N_TOPICS = None  # None = every topic; or an int to cap
N_WORDS = 8
N_COLS = 3           # fewer columns -> wider subplots, titles fit


def sizes_on_disk():
    out = []
    for d in sorted(OUTPUT_DIR.glob("mts_*"), key=lambda p: int(p.name.split("_")[1])):
        if (d / "topics.csv").exists():
            out.append((int(d.name.split("_")[1]), d))
    return out


def combined_csv(sizes):
    frames = []
    for mts, d in sizes:
        df = pd.read_csv(d / "topics.csv")
        df.insert(0, "min_topic_size", mts)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    path = OUTPUT_DIR / "topics_all_sizes.csv"
    combined.to_csv(path, index=False)
    return path, len(combined)


def labeled_barchart(mts, d):
    df = pd.read_csv(d / "topics.csv")
    df = df[df["topic_id"] != -1].sort_values("size", ascending=False)
    if TOP_N_TOPICS is not None:
        df = df.head(TOP_N_TOPICS)

    n = len(df)
    n_rows = (n + N_COLS - 1) // N_COLS
    titles = [f"{row.topic_id}· {str(row.llm_label)[:46]} ({int(row.size)})"
              for row in df.itertuples()]
    fig = make_subplots(rows=n_rows, cols=N_COLS, subplot_titles=titles,
                        horizontal_spacing=0.16, vertical_spacing=0.35 / n_rows)

    palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]
    for i, row in enumerate(df.itertuples()):
        words = [w.strip() for w in str(row.keywords).split(",") if w.strip()][:N_WORDS]
        words = words[::-1]                                  # highest at top
        scores = list(range(1, len(words) + 1))              # rank-based bar length
        r, c = i // N_COLS + 1, i % N_COLS + 1
        fig.add_trace(
            go.Bar(x=scores, y=words, orientation="h",
                   marker_color=palette[i % len(palette)], showlegend=False),
            row=r, col=c,
        )

    fig.update_layout(
        title=f"Topic keywords by LLM label — min_topic_size={mts} ({n} topics)",
        template="plotly_white", height=300 * n_rows, width=1300,
        margin=dict(l=10, r=10, t=90, b=10),
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    for ann in fig.layout.annotations:                       # subplot titles
        ann.font.size = 12
    path = d / "barchart_labeled.html"
    fig.write_html(path)
    return path


def _arr(v):
    """Decode a plotly value that may be a base64 binary array or a plain list."""
    if isinstance(v, dict) and "bdata" in v:
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=v.get("dtype", "f8"))
    return np.asarray(v)


def labeled_documents(mts, d):
    """Rebuild the doc scatter with LLM labels, reusing 2D coords from documents.html."""
    src = d / "documents.html"
    if not src.exists():
        return None
    html = src.read_text()
    m = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*(\[.*?\])\s*,\s*\{', html, re.S)
    if not m:
        return None
    traces = json.loads(m.group(1))

    id2label = dict(zip(pd.read_csv(d / "topics.csv")["topic_id"],
                        pd.read_csv(d / "topics.csv")["llm_label"]))

    rows = []
    for t in traces:
        if "x" not in t or "y" not in t:
            continue
        x, y = _arr(t["x"]), _arr(t["y"])
        hover = t.get("hovertext") or [None] * len(x)
        name = str(t.get("name", ""))
        head = name.split("_", 1)[0]
        tid = int(head) if head.lstrip("-").isdigit() else None
        label = ("other" if tid is None else f"{tid}· {id2label.get(tid, name)}")
        for xi, yi, hi in zip(x, y, hover):
            if hi is None:               # per-topic centroid label marker, not a real doc
                continue
            rows.append((float(xi), float(yi), hi, tid if tid is not None else 1_000_000, label))

    plot_df = (pd.DataFrame(rows, columns=["x", "y", "title", "order", "topic"])
               .sort_values("order"))
    order = plot_df.drop_duplicates("topic")["topic"].tolist()

    fig = px.scatter(plot_df, x="x", y="y", color="topic", hover_name="title",
                     category_orders={"topic": order},
                     title=f"Articles by LLM topic label — min_topic_size={mts} "
                           f"({plot_df['topic'].nunique()} groups)")
    fig.update_traces(marker=dict(size=6, opacity=0.75))
    fig.update_layout(
        template="plotly_white", height=900, width=1400,
        legend=dict(font=dict(size=10), itemsizing="constant",
                    title="topic", tracegroupgap=0),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    path = d / "documents_labeled.html"
    fig.write_html(path)
    return path


def main():
    sizes = sizes_on_disk()
    csv_path, n_rows = combined_csv(sizes)
    print(f"combined: {csv_path}  ({n_rows} rows across {len(sizes)} sizes)")
    for mts, d in sizes:
        print(f"barchart:  {labeled_barchart(mts, d)}")
        print(f"documents: {labeled_documents(mts, d)}")


if __name__ == "__main__":
    main()

"""Single tabbed HTML that gathers EVERY per-size view into one place.

Disk-only. Writes the ambiguity figures as standalone files, then builds
output/report.html which tabs across all views via lazy-loaded iframes:

  Top tabs:  Summary | mts_3 | mts_5 | mts_8 | mts_15
  Per size:  Keywords (barchart) | Documents | Per-topic | Top pairs | Heatmap

Existing files reused as-is: mts_<k>/barchart_labeled.html, mts_<k>/documents_labeled.html
New files written here:       mts_<k>/ambiguity_{per_topic,pairs,heatmap}.html, ambiguity_summary.html

"Secondary topic" = the 2nd-best topic in an article's distribution has p2 > THRESH.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
THRESH = 0.1
PALETTE = {"with": "#EF553B", "without": "#636EFA"}


def sizes_on_disk():
    out = []
    for d in sorted(OUTPUT_DIR.glob("mts_*"), key=lambda p: int(p.name.split("_")[1])):
        if (d / "topic_distr.npy").exists():
            out.append(int(d.name.split("_")[1]))
    return out


def load_size(mts):
    d = OUTPUT_DIR / f"mts_{mts}"
    td = np.load(d / "topic_distr.npy")
    tcsv = pd.read_csv(d / "topics.csv")
    id2label = {int(k): str(v) for k, v in zip(tcsv["topic_id"], tcsv["llm_label"])}
    order = np.argsort(-td, axis=1)[:, :2]
    rng = np.arange(len(td))
    p1 = td[rng, order[:, 0]]
    p2 = td[rng, order[:, 1]]
    return {
        "n": len(td), "id2label": id2label,
        "primary": order[:, 0], "secondary": order[:, 1],
        "p1": p1, "p2": p2, "margin": p1 - p2,
        "has_sec": p2 > THRESH,
    }


def fig_per_topic(s):
    ids = sorted(s["id2label"])
    rows = []
    for t in ids:
        mask = s["primary"] == t
        total = int(mask.sum())
        withs = int((mask & s["has_sec"]).sum())
        rows.append((t, total, withs, total - withs))
    df = pd.DataFrame(rows, columns=["t", "total", "with", "without"]).sort_values("total")
    df = df[df["total"] > 0]
    ylab = [f"{t}· {s['id2label'][t][:34]}" for t in df["t"]]
    share = np.where(df["total"] > 0, df["with"] / df["total"], 0)

    fig = go.Figure()
    fig.add_bar(y=ylab, x=df["without"], name="single topic", orientation="h",
                marker_color=PALETTE["without"], customdata=df["total"],
                hovertemplate="single: %{x} of %{customdata}<extra></extra>")
    fig.add_bar(y=ylab, x=df["with"], name="has secondary", orientation="h",
                marker_color=PALETTE["with"],
                customdata=np.stack([df["total"], share * 100], axis=1),
                hovertemplate="secondary: %{x} of %{customdata[0]} (%{customdata[1]:.0f}%)<extra></extra>")
    n_with = int(s["has_sec"].sum())
    fig.update_layout(
        barmode="stack", template="plotly_white", autosize=True,
        height=max(420, 24 * len(df) + 120),
        title=f"Articles per primary topic — with vs without a secondary topic (p2>{THRESH})<br>"
              f"<sup>overall {n_with}/{s['n']} ({n_with / s['n']:.0%}) have a secondary topic</sup>",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        margin=dict(l=10, r=20, t=80, b=30))
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    fig.update_xaxes(title="article count")
    return fig


def fig_pairs(s, top_n=30):
    pairs = Counter()
    for t1, t2, ok in zip(s["primary"], s["secondary"], s["has_sec"]):
        if ok:
            pairs[(min(int(t1), int(t2)), max(int(t1), int(t2)))] += 1
    if not pairs:
        return go.Figure().update_layout(title="No intersecting pairs above threshold")
    top = pairs.most_common(top_n)[::-1]
    lab = [f"{a}·{s['id2label'][a][:20]}  ↔  {b}·{s['id2label'][b][:20]}" for (a, b), _ in top]
    cnt = [c for _, c in top]
    fig = go.Figure(go.Bar(y=lab, x=cnt, orientation="h", marker_color="#AB63FA",
                           text=cnt, textposition="outside",
                           hovertemplate="%{y}<br>%{x} articles<extra></extra>"))
    fig.update_layout(template="plotly_white", autosize=True,
                      height=max(420, 26 * len(top) + 120),
                      title=f"Top {len(top)} intersecting topic pairs (articles with p2>{THRESH})",
                      margin=dict(l=10, r=40, t=70, b=30))
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    fig.update_xaxes(title="shared article count")
    return fig


def fig_heatmap(s):
    """Color = mean (p1-p2) margin per topic pair. Smaller margin = the two topics are
    nearly tied for an article = a genuinely ambiguous pair. Count kept in the hover."""
    ids = sorted(s["id2label"])
    idx = {t: i for i, t in enumerate(ids)}
    n = len(ids)
    msum = np.zeros((n, n))
    cnt = np.zeros((n, n), dtype=int)
    for t1, t2, mg, ok in zip(s["primary"], s["secondary"], s["margin"], s["has_sec"]):
        if ok:
            a, b = idx[int(t1)], idx[int(t2)]
            for x, y in ((a, b), (b, a)):
                msum[x, y] += float(mg)
                cnt[x, y] += 1
    mean = np.full((n, n), np.nan)
    nz = cnt > 0
    mean[nz] = msum[nz] / cnt[nz]

    labels = [s["id2label"][t][:30] for t in ids]
    cd = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            tag = "" if cnt[i, j] == 0 else f"<br>shared: {cnt[i, j]}<br>mean margin: {mean[i, j]:.2f}"
            cd[i, j] = f"{ids[i]}· {labels[i]}<br>{ids[j]}· {labels[j]}{tag}"
    fig = go.Figure(go.Heatmap(
        z=mean, x=[str(t) for t in ids], y=[str(t) for t in ids],
        customdata=cd, colorscale="YlOrRd", reversescale=True,  # dark = small margin = ambiguous
        zmin=0, hoverongaps=False,
        colorbar=dict(title="mean<br>p1−p2"),
        hovertemplate="%{customdata}<extra></extra>"))
    fig.update_layout(template="plotly_white", autosize=True, height=max(500, 16 * n + 140),
                      title=f"Topic-pair ambiguity — mean (p1−p2) margin, darker = more split "
                            f"(pairs with p2>{THRESH})",
                      margin=dict(l=10, r=10, t=70, b=10))
    fig.update_xaxes(title="secondary topic id", tickfont=dict(size=9))
    fig.update_yaxes(title="primary topic id", autorange="reversed", tickfont=dict(size=9))
    return fig


def fig_summary(states):
    sizes = list(states)
    with_c = [int(states[m]["has_sec"].sum()) for m in sizes]
    n = [states[m]["n"] for m in sizes]
    without_c = [n[i] - with_c[i] for i in range(len(sizes))]
    share = [with_c[i] / n[i] * 100 for i in range(len(sizes))]
    x = [f"mts_{m}" for m in sizes]
    fig = go.Figure()
    fig.add_bar(x=x, y=without_c, name="single topic", marker_color=PALETTE["without"],
                text=without_c, textposition="inside")
    fig.add_bar(x=x, y=with_c, name="has secondary", marker_color=PALETTE["with"],
                text=[f"{with_c[i]}<br>{share[i]:.0f}%" for i in range(len(sizes))],
                textposition="inside")
    fig.update_layout(barmode="stack", template="plotly_white", autosize=True, height=520,
                      title=f"Share & count of articles with a secondary topic (p2>{THRESH}) — by granularity",
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                      margin=dict(l=10, r=10, t=70, b=30))
    fig.update_yaxes(title=f"article count (of {n[0]})")
    return fig


def write_standalone(fig, path):
    fig.write_html(path, include_plotlyjs="cdn", full_html=True,
                   default_height="100%", config={"responsive": True})


def tabbed_html(title, subtitle, tabs, frame_height="86vh"):
    """A flat tabbed page; each tab lazy-loads one iframe. tabs = [(id, label, src), ...]."""
    buttons = "".join(
        f'<button class="tab" id="tab-{tid}" onclick="show(\'{tid}\')">{label}</button>'
        for tid, label, _ in tabs)
    panels = "".join(
        f'<div class="panel" id="panel-{tid}">'
        f'<iframe class="frame" data-src="{src}" loading="lazy"></iframe></div>'
        for tid, _, src in tabs)
    first = tabs[0][0]
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  html,body {{ height:100%; margin:0; font-family:system-ui,sans-serif; background:#fafafa; color:#222; }}
  h1 {{ font-size:17px; margin:10px 16px 2px; }}
  .sub {{ margin:0 16px 8px; color:#666; font-size:12px; }}
  .tabbar {{ display:flex; gap:4px; flex-wrap:wrap; padding:6px 12px; background:#eef; }}
  button {{ border:1px solid #ccd; background:#fff; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:13px; }}
  button.active {{ background:#4458d6; color:#fff; border-color:#4458d6; }}
  .panel {{ display:none; }}
  .frame {{ width:100%; height:{frame_height}; border:1px solid #e2e2ee; background:#fff; }}
</style></head><body>
<h1>{title}</h1>
<div class="sub">{subtitle}</div>
<div class="tabbar">{buttons}</div>
{panels}
<script>
function show(id){{
  document.querySelectorAll('.panel').forEach(function(p){{ p.style.display='none'; }});
  document.querySelectorAll('.tab').forEach(function(t){{ t.classList.remove('active'); }});
  var panel = document.getElementById('panel-'+id);
  panel.style.display='block';
  document.getElementById('tab-'+id).classList.add('active');
  var f = panel.querySelector('iframe');
  if (f && !f.src && f.dataset.src) f.src = f.dataset.src;
}}
show('{first}');
</script>
</body></html>"""


def build():
    sizes = sizes_on_disk()
    states = {m: load_size(m) for m in sizes}

    # 1) ambiguity figures as standalone files
    write_standalone(fig_summary(states), OUTPUT_DIR / "ambiguity_summary.html")
    for m in sizes:
        d = OUTPUT_DIR / f"mts_{m}"
        write_standalone(fig_per_topic(states[m]), d / "ambiguity_per_topic.html")
        write_standalone(fig_pairs(states[m]), d / "ambiguity_pairs.html")
        write_standalone(fig_heatmap(states[m]), d / "ambiguity_heatmap.html")

    # 2) one combined report PER mts (all views tabbed; srcs are local to mts_<k>/)
    for m in sizes:
        d = OUTPUT_DIR / f"mts_{m}"
        views = [
            ("barchart", "Keywords", "barchart_labeled.html"),
            ("documents", "Documents", "documents_labeled.html"),
            ("pertopic", "Per-topic", "ambiguity_per_topic.html"),
            ("pairs", "Top pairs", "ambiguity_pairs.html"),
            ("heatmap", "Heatmap", "ambiguity_heatmap.html"),
        ]
        sub = (f'min_topic_size={m}. "Secondary topic" = 2nd-best topic in an article\'s '
               f'distribution has p2 &gt; {THRESH}.')
        (d / "report.html").write_text(
            tabbed_html(f"mts_{m} — topic report", sub, views, "82vh"), encoding="utf-8")

    # 3) master report: one tab per mts (iframes its per-mts report) + a Summary tab
    master = [("summary", "Summary", "ambiguity_summary.html")]
    master += [(f"mts{m}", f"mts_{m}", f"mts_{m}/report.html") for m in sizes]
    sub = ('Each granularity tab is its full per-mts report. '
           f'"Secondary topic" = 2nd-best topic has p2 &gt; {THRESH}.')
    out = OUTPUT_DIR / "report.html"
    out.write_text(tabbed_html("Topic report — all granularities", sub, master, "88vh"),
                   encoding="utf-8")
    print(f"wrote {out} + {len(sizes)} per-mts report.html  (sizes: {sizes})")


if __name__ == "__main__":
    build()

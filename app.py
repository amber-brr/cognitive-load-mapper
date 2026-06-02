import os

from dotenv import load_dotenv
load_dotenv()

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8080")

# Shapes that appear as article labels (data-driven, from cluster_shapes.py)
ARTICLE_SHAPES = ["plateau", "rollercoaster", "resolution"]

# All shapes available as rewrite targets (includes ramp/cliff as aspirational targets)
REWRITE_SHAPES = ["ramp", "cliff", "plateau", "rollercoaster", "resolution"]

SHAPE_DESC = {
    "ramp": "Complexity builds toward the end",
    "cliff": "Dense opening, simpler after",
    "plateau": "Consistent complexity throughout",
    "rollercoaster": "Alternating high and low complexity",
    "resolution": "Complexity fades toward the end",
}

PUBLICATIONS = {
    "www.construction-physics.com":    "Construction Physics",
    "www.experimental-history.com":    "Experimental History",
    "www.oneusefulthing.org":          "One Useful Thing",
    "www.theintrinsicperspective.com": "The Intrinsic Perspective",
    "www.ageofinvention.xyz":          "Age of Invention",
}


def api_get(path, params=None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is it running at " + API_BASE + "?")
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e}")
        st.stop()


def api_post(path, body):
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=180)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is it running at " + API_BASE + "?")
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e}")
        st.stop()


def count_flagged(paras, target_shape):
    """Mirrors the server-side flag_paragraphs logic to preview the flagged count."""
    if not paras:
        return 0
    scores = [p["complexity_v1"] for p in paras]
    mean_s = sum(scores) / len(scores)
    variance = sum((s - mean_s) ** 2 for s in scores) / len(scores)
    std_s = variance ** 0.5 or 1.0
    n = len(paras)
    flagged = []

    if target_shape == "ramp":
        for p in paras:
            pos, c = p.get("paragraph_position_norm", 0), p["complexity_v1"]
            if pos < 0.4 and c > mean_s:
                flagged.append(p)
            elif pos > 0.6 and c < mean_s:
                flagged.append(p)
    elif target_shape == "resolution":
        for p in paras:
            pos, c = p.get("paragraph_position_norm", 0), p["complexity_v1"]
            if pos > 0.7 and c > mean_s:
                flagged.append(p)
    elif target_shape == "cliff":
        for p in paras:
            pos, c = p.get("paragraph_position_norm", 0), p["complexity_v1"]
            if pos > 0.2 and c > mean_s + 0.5 * std_s:
                flagged.append(p)
    elif target_shape == "plateau":
        for p in paras:
            if abs(p["complexity_v1"] - mean_s) > std_s:
                flagged.append(p)
    elif target_shape == "rollercoaster":
        signs = [
            1 if p["complexity_v1"] > mean_s + 0.1 * std_s
            else (-1 if p["complexity_v1"] < mean_s - 0.1 * std_s else 0)
            for p in paras
        ]
        for i in range(1, n - 1):
            if signs[i] != 0 and signs[i - 1] == signs[i] == signs[i + 1]:
                flagged.append(paras[i])

    flagged.sort(key=lambda p: abs(p["complexity_v1"] - mean_s), reverse=True)
    return min(len(flagged), 3)


# ── Page setup ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="CogLoad Mapper", layout="wide")
st.title("Cognitive Load Mapper")
st.caption("Visualize how cognitive load flows through an article.")

# ── Sidebar: browse and filter ───────────────────────────────────────────────
with st.sidebar:
    st.header("Browse Articles")
    selected_pub = st.radio(
        "Publication",
        options=list(PUBLICATIONS.keys()),
        format_func=lambda k: PUBLICATIONS[k],
    )
    shape_filter = st.selectbox("Filter by shape", ["(any)"] + ARTICLE_SHAPES)

    with st.expander("What do shapes mean?"):
        for shape in ARTICLE_SHAPES:
            st.markdown(f"**{shape.capitalize()}** — {SHAPE_DESC[shape]}")

    shape_q = None if shape_filter == "(any)" else shape_filter

    articles = api_get("/articles", params={"publication": selected_pub, "shape": shape_q, "limit": 100})

    if not articles:
        st.info("No articles match your filters.")
        st.stop()

    article_labels = {
        a["article_id"]: a["title"] or "Untitled"
        for a in articles
    }
    selected_id = st.selectbox(
        "Select article",
        options=list(article_labels.keys()),
        format_func=lambda i: article_labels[i],
    )

# ── Article header ───────────────────────────────────────────────────────────
article = api_get(f"/articles/{selected_id}")

st.subheader(article["title"] or "Untitled")
st.caption(
    f"{article['publication_name']} · {article.get('publish_date') or ''}"
    f" · [Read original]({article['post_url']})"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Gradient Shape", article["gradient_shape"])
col2.metric("Word Count", article.get("word_count") or "—")
col3.metric("Mean Complexity", f"{article['mean_complexity']:.2f}")
engagement = article.get("engagement_z")
col4.metric(
    "Engagement Score",
    f"{engagement:.2f}" if engagement is not None else "—",
    help="Z-score vs. the publication average — positive means higher engagement than typical for this source.",
)
st.caption(SHAPE_DESC.get(article["gradient_shape"], ""))

# ── Full article text ────────────────────────────────────────────────────────
paras = article.get("paragraphs", [])
if paras:
    with st.expander("Full Article Text", expanded=False):
        full_text = "\n\n".join(p["paragraph_text"] for p in paras)
        st.markdown(full_text)

# ── Complexity trajectory ────────────────────────────────────────────────────
if paras:
    mean_c = article["mean_complexity"]

    df_chart = pd.DataFrame({
        "Paragraph": [p["paragraph_index"] + 1 for p in paras],
        "Complexity": [p["complexity_v1"] for p in paras],
    })

    line = alt.Chart(df_chart).mark_line(point=True, color="#4C78A8").encode(
        x=alt.X("Paragraph:Q", title="Paragraph"),
        y=alt.Y("Complexity:Q", title="Complexity"),
        tooltip=["Paragraph:Q", alt.Tooltip("Complexity:Q", format=".2f")],
    )
    mean_rule = alt.Chart(
        pd.DataFrame({"Mean": [mean_c]})
    ).mark_rule(color="red", strokeDash=[6, 4], opacity=0.6).encode(
        y="Mean:Q",
        tooltip=[alt.Tooltip("Mean:Q", title="Mean complexity", format=".2f")],
    )

    st.subheader("Complexity Trajectory")
    st.altair_chart((line + mean_rule).properties(height=300), width='stretch')
    st.caption("Red dashed line = mean complexity")

    # ── Paragraph browser ────────────────────────────────────────────────────
    st.subheader("Paragraphs")
    for p in paras:
        above = p["complexity_v1"] > mean_c
        icon = "🔴" if above else "🟢"
        tag = "↑ above avg" if above else "↓ below avg"
        label = f"{icon} §{p['paragraph_index'] + 1}  ·  complexity {p['complexity_v1']:.2f}  ·  {tag}"
        with st.expander(label):
            st.write(p["paragraph_text"])

# ── Rewrite section ──────────────────────────────────────────────────────────
st.divider()
st.subheader("Rewrite toward a target shape")

current_shape = article["gradient_shape"]
other_shapes = [s for s in REWRITE_SHAPES if s != current_shape]
target_shape = st.selectbox("Target shape", other_shapes)
st.caption(
    f"**{current_shape.capitalize()}** → **{target_shape.capitalize()}** "
    f"— {SHAPE_DESC.get(target_shape, '')}"
)

if paras:
    n_flagged = count_flagged(paras, target_shape)
    if n_flagged == 0:
        st.info("No paragraphs need rewriting for this target shape.")
    else:
        st.caption(f"{n_flagged} paragraph{'s' if n_flagged != 1 else ''} will be targeted for rewriting.")

if st.button("Rewrite flagged paragraphs", type="primary"):
    with st.spinner("Sending to LLM — this may take up to 3 minutes…"):
        result = api_post(f"/articles/{selected_id}/rewrite", {"target_shape": target_shape})

    if result.get("message"):
        st.info(result["message"])
    elif not result.get("flagged_paragraphs"):
        st.info("No paragraphs needed rewriting.")
    else:
        st.success(f"{len(result['flagged_paragraphs'])} paragraph(s) rewritten.")
        for fp in result["flagged_paragraphs"]:
            st.markdown(f"**§{fp['paragraph_index'] + 1}** — _{fp['reason']}_")
            left, right = st.columns(2)
            left.text_area(
                "Original", fp["paragraph_text"],
                height=160, disabled=True, key=f"orig_{fp['paragraph_index']}",
            )
            right.text_area(
                "Rewritten", fp["rewritten_text"],
                height=160, disabled=True, key=f"new_{fp['paragraph_index']}",
            )

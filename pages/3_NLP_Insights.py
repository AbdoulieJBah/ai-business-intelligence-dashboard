import streamlit as st
import pandas as pd
import plotly.express as px

from utils import inject_css, section_title, card_open, card_close, apply_plotly_theme, build_tfidf_keywords, cluster_reviews, get_word_counts, basic_stopwords
if "dashboard_df_raw" not in st.session_state:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-title">🧠 No text dataset loaded yet</div>
        <div class="empty-text">
            Upload your dataset in the Overview page first to explore sentiment, keywords, and review clusters.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Go to Overview"):
        st.switch_page("pages/1_Overview.py")
    st.stop()

st.set_page_config(page_title="NLP Insights", page_icon="🧠", layout="wide")
inject_css()

if "dashboard_df_raw" not in st.session_state:
    st.warning("Go to the Overview page first and upload a dataset.")
    st.stop()

df_raw = st.session_state["dashboard_df_raw"]
text_col = st.session_state.get("dashboard_text_col")
title_col = st.session_state.get("dashboard_title_col")

if text_col is None:
    st.info("This dataset does not contain review or text columns for NLP analysis.")
    st.stop()

card_open()
section_title("🧠 NLP Insights", "Sentiment, keywords, clusters, and review exploration.")
n1, n2, n3 = st.columns(3)
n1.metric("Total Reviews", len(df_raw))
n2.metric("Average Review Length", f"{df_raw['review_length'].mean():.1f} words" if "review_length" in df_raw.columns else "N/A")
n3.metric("Average Rating", f"{df_raw['rating'].mean():.2f}" if "rating" in df_raw.columns else "N/A")
card_close()

if "sentiment_label" in df_raw.columns:
    card_open()
    section_title("🙂 Sentiment Summary")
    sentiment_counts = df_raw["sentiment_label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    fig = px.bar(sentiment_counts, x="Sentiment", y="Count", title="Sentiment Summary")
    fig = apply_plotly_theme(fig, 320)
    st.plotly_chart(fig, use_container_width=True)
    card_close()

card_open()
section_title("🔑 TF-IDF Keywords")
keyword_df, _, _ = build_tfidf_keywords(df_raw["clean_text"], top_n=15)
if not keyword_df.empty:
    fig = px.bar(keyword_df, x="Score", y="Keyword", orientation="h", title="Top Keywords")
    fig = apply_plotly_theme(fig, 420)
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough text data to extract keywords.")
card_close()

card_open()
section_title("🧩 Topic Clusters")
cluster_count = st.slider("Number of topic clusters", min_value=2, max_value=6, value=3)
clustered_df, top_terms_per_cluster, _ = cluster_reviews(df_raw["clean_text"], n_clusters=cluster_count)

if clustered_df is not None:
    cluster_summary = clustered_df["cluster"].value_counts().sort_index().reset_index()
    cluster_summary.columns = ["Cluster", "Count"]
    fig = px.bar(cluster_summary, x="Cluster", y="Count", title="Topic Clusters")
    fig = apply_plotly_theme(fig, 320)
    st.plotly_chart(fig, use_container_width=True)

    for cluster_id, terms in top_terms_per_cluster.items():
        st.markdown(f"<div class='info-card'><strong>Cluster {cluster_id}:</strong> {', '.join(terms)}</div>", unsafe_allow_html=True)
else:
    st.info("Not enough text data to build topic clusters.")
card_close()

card_open()
section_title("➕ / ➖ Positive vs Negative Keywords")
if "sentiment_label" in df_raw.columns:
    positive_df = df_raw[df_raw["sentiment_label"] == "Positive"]
    negative_df = df_raw[df_raw["sentiment_label"] == "Negative"]

    pos_words = get_word_counts(positive_df["clean_text"], stopwords=basic_stopwords, top_n=10)
    neg_words = get_word_counts(negative_df["clean_text"], stopwords=basic_stopwords, top_n=10)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top Positive Words**")
        st.dataframe(pd.DataFrame(pos_words, columns=["Word", "Count"]), use_container_width=True)
    with c2:
        st.markdown("**Top Negative Words**")
        st.dataframe(pd.DataFrame(neg_words, columns=["Word", "Count"]), use_container_width=True)
card_close()

card_open()
section_title("🔍 Search Reviews")
query = st.text_input("Search for a keyword in reviews")
if query:
    matched = df_raw[df_raw["clean_text"].str.contains(query.lower(), na=False)]
    if title_col is not None:
        st.dataframe(matched[[title_col, text_col]].head(20), use_container_width=True)
    else:
        st.dataframe(matched[[text_col]].head(20), use_container_width=True)
card_close()

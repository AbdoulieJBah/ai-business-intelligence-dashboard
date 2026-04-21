import streamlit as st

st.set_page_config(
    page_title="Universal AI Business Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
:root {
    --bg: #07111f;
    --panel: #0f172a;
    --panel-2: #111827;
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #3b82f6;
    --border: rgba(148, 163, 184, 0.14);
    --shadow: 0 16px 40px rgba(0, 0, 0, 0.28);
}
.stApp {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.12), transparent 28%),
        linear-gradient(180deg, #07111f 0%, #0b1220 100%);
    color: var(--text);
}
.block-container {
    max-width: 1450px;
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}
.hero-wrap {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.22), transparent 30%),
        linear-gradient(135deg, rgba(17,24,39,0.98), rgba(15,23,42,0.98));
    border: 1px solid rgba(59,130,246,0.20);
    border-radius: 24px;
    padding: 30px;
    box-shadow: var(--shadow);
    margin-bottom: 18px;
}
.hero-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(59,130,246,0.16);
    color: #bfdbfe;
    font-size: 0.82rem;
    font-weight: 700;
    margin-bottom: 10px;
}
.info-card {
    background: linear-gradient(180deg, rgba(30,41,59,0.95), rgba(15,23,42,0.98));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.20);
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">AI Powered Analytics Suite</div>
    <h1 style="margin:0;">📊 Universal AI Business Dashboard</h1>
    <p style="font-size:1.1rem; color:#dbeafe; margin:8px 0;">
        Forecasting • Anomaly Detection • NLP Insights • Interactive Analytics
    </p>
    <p style="color:#94a3b8;">
        Upload sales, finance, or customer review datasets and turn raw data into decision-ready insights.
    </p>
</div>
""", unsafe_allow_html=True)

st.info("Use the sidebar to navigate between pages after uploading your dataset in the Overview page.")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("<div class='info-card'><strong>Overview</strong><br>Upload files, detect metrics, and explore KPIs.</div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='info-card'><strong>Forecasting</strong><br>Trend estimation with linear regression and moving average.</div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='info-card'><strong>NLP Insights</strong><br>Sentiment, keywords, clusters, and review exploration.</div>", unsafe_allow_html=True)

st.markdown("### Navigation")
st.write("Go to **Overview** first, upload your file, then move through the other pages.")

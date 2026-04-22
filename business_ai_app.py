import streamlit as st
from utils import inject_css

st.set_page_config(
    page_title="Universal AI Business Dashboard",
    page_icon="📊",
    layout="wide"
)

inject_css()

st.markdown("""
<div class="hero-card">
    <div class="hero-badge">AI Powered Analytics Suite</div>
    <div class="hero-title">📊 Universal AI Business Dashboard</div>
    <div class="hero-subtitle">
        Forecasting • Anomaly Detection • NLP Insights • Interactive Analytics
    </div>
    <div class="hero-muted">
        Upload sales, finance, or customer review datasets and turn raw data into decision-ready insights.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    Use the buttons below to navigate between pages after uploading your dataset in the Overview page.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("## Dashboard Sections")

# -------- OVERVIEW --------
st.markdown("""
<div class="mini-card">
    <h3 style="margin-top:0; color:#f8fafc;">📊 Overview</h3>
    <p style="color:#cbd5e1;">Upload files, detect metrics, and explore KPIs.</p>
</div>
""", unsafe_allow_html=True)

if st.button("Open Overview", key="go_overview", width="stretch"):
    st.switch_page("pages/1_Overview.py")

st.markdown("<br>", unsafe_allow_html=True)

# -------- FORECASTING --------
st.markdown("""
<div class="mini-card">
    <h3 style="margin-top:0; color:#f8fafc;">🔮 Forecasting</h3>
    <p style="color:#cbd5e1;">Trend estimation with linear regression and moving average.</p>
</div>
""", unsafe_allow_html=True)

if st.button("Open Forecasting", key="go_forecasting", width="stretch"):
    st.switch_page("pages/2_Forecasting.py")

st.markdown("<br>", unsafe_allow_html=True)

# -------- NLP --------
st.markdown("""
<div class="mini-card">
    <h3 style="margin-top:0; color:#f8fafc;">🧠 NLP Insights</h3>
    <p style="color:#cbd5e1;">Sentiment, keywords, clusters, and review exploration.</p>
</div>
""", unsafe_allow_html=True)

if st.button("Open NLP Insights", key="go_nlp", width="stretch"):
    st.switch_page("pages/3_NLP_Insights.py")

st.markdown("<br>", unsafe_allow_html=True)

# -------- TABLES --------
st.markdown("""
<div class="mini-card">
    <h3 style="margin-top:0; color:#f8fafc;">📥 Tables & Downloads</h3>
    <p style="color:#cbd5e1;">Inspect processed tables and export clean reports.</p>
</div>
""", unsafe_allow_html=True)

if st.button("Open Tables & Downloads", key="go_tables", width="stretch"):
    st.switch_page("pages/4_Tables_&_Downloads.py")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="
    text-align:center;
    margin-top:30px;
    padding:18px;
    border-radius:16px;
    background:linear-gradient(180deg, rgba(30,41,59,0.7), rgba(15,23,42,0.9));
    border:1px solid rgba(59,130,246,0.25);
">

    <div style="
        font-size:1.05rem;
        color:#e5e7eb;
        margin-bottom:6px;
    ">
        Built by <strong style="color:#3b82f6;">Abdoulie J Bah</strong> 🚀
    </div>

    <div style="
        font-size:0.9rem;
        color:#94a3b8;
        margin-bottom:12px;
    ">
        AI Engineer • Data Scientist • Business Intelligence Developer
    </div>

    <div style="display:flex; justify-content:center; gap:10px; flex-wrap:wrap;">

        <a href="https://www.linkedin.com/in/abdoulie-j-bah-b71263244" target="_blank" style="
            text-decoration:none;
            padding:8px 14px;
            border-radius:10px;
            background:#0ea5e9;
            color:white;
            font-weight:600;
        ">LinkedIn</a>

        <a href="https://github.com/AbdoulieJBah/ai-business-intelligence-dashboard" target="_blank" style="
            text-decoration:none;
            padding:8px 14px;
            border-radius:10px;
            background:#1f2937;
            color:white;
            font-weight:600;
            border:1px solid rgba(255,255,255,0.1);
        ">GitHub</a>

        <a href="mailto:21722285bah@gmail.com" style="
            text-decoration:none;
            padding:8px 14px;
            border-radius:10px;
            background:#2563eb;
            color:white;
            font-weight:600;
        ">Contact</a>

    </div>

</div>
""", unsafe_allow_html=True)

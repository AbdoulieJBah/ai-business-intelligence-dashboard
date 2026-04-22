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
</div>
""", unsafe_allow_html=True)

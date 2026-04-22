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

st.markdown("## Dashboard Sections")

# Mobile-friendly: stacked layout
st.markdown("""
<div class="mini-card">
    <h3 style="margin-top:0; color:#f8fafc;">Overview</h3>
    <p style="color:#cbd5e1; margin-bottom:0;">
        Upload files, detect metrics, and explore KPIs.
    </p>
</div>
""", unsafe_allow_html=True)
if st.button("Open Overview", key="go_overview", width="stretch"):
    st.switch_page("pages/1_Overview.py")

st.markdown("""
<div class="mini-card">
    <h3 style="margin-top:0; color:#f8fafc;">Forecasting</h3>
    <p style="color:#cbd5e1; margin-bottom:0;">
        Trend estimation with linear regression and moving average.
    </p>
</div>
""", unsafe_allow_html=True)
if st.button("Open Forecasting", key="go_forecasting", width="stretch"):
    st.switch_page("pages/2_Forecasting.py")

st.markdown("""
<div class="mini-card">
    <h3 style="margin-top:0; color:#f8fafc;">NLP Insights</h3>
    <p style="color:#cbd5e1; margin-bottom:0;">
        Sentiment, keywords, clusters, and review exploration.
    </p>
</div>
""", unsafe_allow_html=True)
if st.button("Open NLP Insights", key="go_nlp", width="stretch"):
    st.switch_page("pages/3_NLP_Insights.py")

st.markdown("""
<div class="mini-card">
    <h3 style="margin-top:0; color:#f8fafc;">Tables & Downloads</h3>
    <p style="color:#cbd5e1; margin-bottom:0;">
        Inspect processed tables and export clean reports.
    </p>
</div>
""", unsafe_allow_html=True)
if st.button("Open Tables & Downloads", key="go_tables", width="stretch"):
    st.switch_page("pages/4_Tables_&_Downloads.py")

st.markdown("## Navigation")

st.markdown("""
<div class="premium-card">
    <div style="color:#cbd5e1; font-size:1.02rem;">
        Go to <strong>Overview</strong> first, upload your file, then move through the other pages.
    </div>
</div>
""", unsafe_allow_html=True)

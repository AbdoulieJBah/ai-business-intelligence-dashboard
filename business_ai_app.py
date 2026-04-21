import streamlit as st
from utils import inject_css

st.set_page_config(page_title="Universal AI Business Dashboard", page_icon="📊", layout="wide")
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

st.markdown("<div class='info-card'>Use the sidebar to navigate between pages after uploading your dataset in the Overview page.</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="mini-card">
        <h4 style="margin-bottom:8px;">Overview</h4>
        <p style="color:#cbd5e1;">Upload files, detect metrics, and explore KPIs.</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="mini-card">
        <h4 style="margin-bottom:8px;">Forecasting</h4>
        <p style="color:#cbd5e1;">Trend estimation with linear regression and moving average.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="mini-card">
        <h4 style="margin-bottom:8px;">NLP Insights</h4>
        <p style="color:#cbd5e1;">Sentiment, keywords, clusters, and review exploration.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("## Navigation")
st.write("Go to **Overview** first, upload your file, then move through the other pages.")

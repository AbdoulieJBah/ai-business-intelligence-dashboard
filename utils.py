import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def inject_css():
    st.markdown("""
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
        border: 1px solid rgba(59,130,246,0.20);
        border-radius: 24px;
        padding: 28px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        margin-bottom: 1rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(59,130,246,0.18);
        color: #93c5fd;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 14px;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.35rem;
        line-height: 1.1;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        margin-bottom: 0.6rem;
    }

    .hero-muted {
        color: #94a3b8;
        font-size: 0.96rem;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 0.4rem;
        margin-bottom: 0.15rem;
    }

    .section-subtitle {
        color: #94a3b8;
        font-size: 0.96rem;
        margin-bottom: 1rem;
    }

    .premium-card {
        background: linear-gradient(180deg, rgba(17,24,39,0.95), rgba(15,23,42,0.95));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        margin-bottom: 1rem;
    }

    .mini-card {
        background: linear-gradient(180deg, rgba(30,41,59,0.75), rgba(17,24,39,0.88));
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 18px;
        min-height: 120px;
    }

    .info-card {
        background: rgba(59,130,246,0.08);
        border: 1px solid rgba(59,130,246,0.18);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
        color: #dbeafe;
    }

    .empty-state {
        background: linear-gradient(180deg, rgba(17,24,39,0.98), rgba(15,23,42,0.98));
        border: 1px dashed rgba(59,130,246,0.30);
        border-radius: 24px;
        padding: 36px 28px;
        text-align: center;
        margin-top: 2rem;
    }

    .empty-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }

    .empty-text {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1rem;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(17,24,39,0.94), rgba(15,23,42,0.94));
        border: 1px solid rgba(255,255,255,0.06);
        padding: 16px;
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.16);
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 14px !important;
    }

    .stButton button {
        border-radius: 14px;
        font-weight: 700;
        padding: 0.6rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


def section_title(title, subtitle=""):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='section-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def card_open():
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)

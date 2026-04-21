import streamlit as st
import plotly.express as px

from utils import (
    inject_css,
    section_title,
    card_open,
    card_close,
    apply_plotly_theme,
    fmt_num
)

st.set_page_config(page_title="Forecasting", page_icon="🔮", layout="wide")
inject_css()

if "dashboard_df" not in st.session_state:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-title">📂 No dataset loaded yet</div>
        <div class="empty-text">
            Upload your file in the Overview page first to unlock forecasting and trend analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Go to Overview"):
        st.switch_page("pages/1_Overview.py")
    st.stop()

df = st.session_state["dashboard_df"]
metric_col = st.session_state["dashboard_metric_col"]
time_col = st.session_state["dashboard_time_col"]
prediction_lr = st.session_state["dashboard_prediction_lr"]
prediction_ma = st.session_state["dashboard_prediction_ma"]
growth_pct = st.session_state["dashboard_growth_pct"]
slope = st.session_state.get("dashboard_slope")

st.markdown("""
<div class="hero-card">
    <div class="hero-badge">Forecasting</div>
    <div class="hero-title">🔮 Trend & Forecast Analysis</div>
    <div class="hero-subtitle">
        Explore time-based movement, short-term forecasting, and metric distribution.
    </div>
    <div class="hero-muted">
        Built using regression-based forecasting and trend analysis from your uploaded dataset.
    </div>
</div>
""", unsafe_allow_html=True)

card_open()
section_title("📈 Forecast Summary", "Core forecasting indicators for the selected metric.")
c1, c2, c3, c4 = st.columns(4)

c1.metric("Linear Forecast", fmt_num(prediction_lr) if prediction_lr is not None else "N/A")
c2.metric("Moving Average", fmt_num(prediction_ma) if prediction_ma is not None else "N/A")
c3.metric("Growth %", f"{growth_pct:.2f}%" if growth_pct is not None else "N/A")

if slope is not None:
    trend_label = "Upward" if slope > 0 else "Downward" if slope < 0 else "Stable"
else:
    trend_label = "N/A"

c4.metric("Trend Direction", trend_label)
card_close()

card_open()
section_title("📊 Metric Trend Over Time", f"Time-series view of {metric_col}.")
trend_df = df[[time_col, metric_col]].copy().sort_values(time_col)

fig = px.line(
    trend_df,
    x=time_col,
    y=metric_col,
    markers=True,
    title=f"{metric_col} over time"
)
fig = apply_plotly_theme(fig, 460)
fig.update_traces(line=dict(width=3), marker=dict(size=5))
fig.update_layout(title_x=0.02)

st.plotly_chart(fig, width="stretch")
card_close()

card_open()
section_title("📉 Metric Distribution", f"Distribution pattern of {metric_col}.")
hist_fig = px.histogram(
    df,
    x=metric_col,
    nbins=25,
    title=f"Distribution of {metric_col}"
)
hist_fig = apply_plotly_theme(hist_fig, 420)
hist_fig.update_layout(title_x=0.02)

st.plotly_chart(hist_fig, width="stretch")
card_close()

card_open()
section_title("🧠 Forecast Interpretation", "What these forecast numbers suggest.")

insights = []

if prediction_lr is not None and prediction_ma is not None:
    if prediction_lr > prediction_ma:
        insights.append("The linear forecast is above the moving average, suggesting possible upward momentum.")
    elif prediction_lr < prediction_ma:
        insights.append("The linear forecast is below the moving average, suggesting potential short-term weakness.")
    else:
        insights.append("The linear forecast and moving average are aligned, indicating a steady outlook.")

if growth_pct is not None:
    if growth_pct > 0:
        insights.append(f"The selected metric has grown by {growth_pct:.2f}% across the observed period.")
    elif growth_pct < 0:
        insights.append(f"The selected metric has declined by {abs(growth_pct):.2f}% across the observed period.")
    else:
        insights.append("The selected metric remained stable across the observed period.")

if slope is not None:
    if slope > 0:
        insights.append("Trend slope is positive, which indicates an overall improving direction.")
    elif slope < 0:
        insights.append("Trend slope is negative, which indicates an overall declining direction.")
    else:
        insights.append("Trend slope is neutral, suggesting flat performance.")

for insight in insights:
    st.markdown(f"<div class='info-card'>{insight}</div>", unsafe_allow_html=True)

card_close()

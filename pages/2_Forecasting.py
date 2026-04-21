import streamlit as st
import plotly.express as px

from utils import inject_css, section_title, card_open, card_close, apply_plotly_theme, fmt_num

st.set_page_config(page_title="Forecasting", page_icon="🔮", layout="wide")
inject_css()

if "dashboard_df" not in st.session_state:
    st.warning("Go to the Overview page first and upload a dataset.")
    st.stop()

df = st.session_state["dashboard_df"]
metric_col = st.session_state["dashboard_metric_col"]
time_col = st.session_state["dashboard_time_col"]
prediction_lr = st.session_state["dashboard_prediction_lr"]
prediction_ma = st.session_state["dashboard_prediction_ma"]
growth_pct = st.session_state["dashboard_growth_pct"]

card_open()
section_title("🔮 Forecasting", "Trend estimation and time-based visualization.")
c1, c2, c3 = st.columns(3)
c1.metric("Linear Forecast", fmt_num(prediction_lr) if prediction_lr is not None else "N/A")
c2.metric("Moving Average Forecast", fmt_num(prediction_ma) if prediction_ma is not None else "N/A")
c3.metric("Growth %", f"{growth_pct:.2f}%" if growth_pct is not None else "N/A")
card_close()

card_open()
trend_df = df[[time_col, metric_col]].copy().sort_values(time_col)
fig = px.line(trend_df, x=time_col, y=metric_col, markers=True, title=f"{metric_col} over time")
fig = apply_plotly_theme(fig, 460)
fig.update_traces(line=dict(width=3), marker=dict(size=5))
st.plotly_chart(fig, use_container_width=True)
card_close()

card_open()
hist_fig = px.histogram(df, x=metric_col, nbins=25, title=f"Distribution of {metric_col}")
hist_fig = apply_plotly_theme(hist_fig, 420)
st.plotly_chart(hist_fig, use_container_width=True)
card_close()

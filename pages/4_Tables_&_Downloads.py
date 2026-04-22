import plotly.express as px
from utils import inject_css, section_title, card_open, card_close, apply_plotly_theme
import streamlit as st
import pandas as pd

from utils import inject_css, section_title, card_open, card_close

st.set_page_config(page_title="Tables & Downloads", page_icon="📥", layout="wide")
inject_css()

if "dashboard_df" not in st.session_state:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-title">📥 No processed data available</div>
        <div class="empty-text">
            Upload a dataset in the Overview page first to access cleaned tables and downloads.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Go to Overview"):
        st.switch_page("pages/1_Overview.py")
    st.stop()

df = st.session_state["dashboard_df"]
metric_col = st.session_state["dashboard_metric_col"]
category_col = st.session_state["dashboard_category_col"]
time_col = st.session_state["dashboard_time_col"]
prediction = st.session_state["dashboard_prediction"]
growth_pct = st.session_state["dashboard_growth_pct"]
profit_col = st.session_state["dashboard_profit_col"]
cost_col = st.session_state["dashboard_cost_col"]
quantity_col = st.session_state["dashboard_quantity_col"]
margin_available = st.session_state["dashboard_margin_available"]

total_metric = st.session_state["dashboard_total_metric"]
avg_metric = st.session_state["dashboard_avg_metric"]
max_metric = df[metric_col].max()
min_metric = df[metric_col].min()

st.markdown("""
<div class="hero-card">
    <div class="hero-badge">Tables & Downloads</div>
    <div class="hero-title">📥 Data Tables & Export Center</div>
    <div class="hero-subtitle">
        Inspect cleaned records, review category summaries, and export reports.
    </div>
    <div class="hero-muted">
        This page gives you direct access to processed data outputs and downloadable files.
    </div>
</div>
""", unsafe_allow_html=True)

card_open()
section_title("📋 Detailed Tables", "Review top records, bottom records, and grouped category summaries.")

left, right = st.columns(2)

with left:
    st.markdown("#### Top 10 Rows by Main Metric")
    st.dataframe(
        df.sort_values(metric_col, ascending=False).head(10),
        width="stretch"
    )

with right:
    st.markdown("#### Bottom 10 Rows by Main Metric")
    st.dataframe(
        df.sort_values(metric_col, ascending=True).head(10),
        width="stretch"
    )

if (
    category_col != "None"
    and category_col in df.columns
    and category_col != metric_col
    and (df[category_col].dtype == "object" or str(df[category_col].dtype).startswith("category"))
):
    st.markdown("#### Category Summary")

    cat_summary = (
        df.groupby(category_col)[metric_col]
        .agg(["sum", "mean", "count"])
        .sort_values("sum", ascending=False)
        .reset_index()
    )

    cat_summary.columns = [
        category_col,
        f"Total {metric_col}",
        f"Average {metric_col}",
        "Records"
    ]

    st.dataframe(cat_summary, width="stretch")

    st.markdown("#### Category Visual Analysis")

    category_chart_df = (
        df.groupby(category_col, as_index=False)[metric_col]
        .sum()
        .sort_values(metric_col, ascending=False)
        .head(10)
    )

    fig = px.bar(
        category_chart_df,
        x=category_col,
        y=metric_col,
        title=f"{metric_col} by {category_col}",
        text_auto=".2s"
    )

    fig = apply_plotly_theme(fig, 420)
    fig.update_layout(
        xaxis_title=category_col,
        yaxis_title=f"Total {metric_col}"
    )

    st.plotly_chart(fig, width="stretch")

elif category_col == metric_col:
    st.warning("Category summary is unavailable because the category and main metric are the same column.")

card_close()

card_open()
section_title("📥 Downloads", "Export cleaned data and summary outputs.")

summary_rows = [
    ["Main Metric", metric_col],
    [f"Total {metric_col}", round(total_metric, 2)],
    [f"Average {metric_col}", round(avg_metric, 2)],
    [f"Highest {metric_col}", round(max_metric, 2)],
    [f"Lowest {metric_col}", round(min_metric, 2)],
    ["Rows after filtering", len(df)]
]

if prediction is not None:
    summary_rows.append([f"Predicted Next {metric_col}", round(prediction, 2)])

if growth_pct is not None:
    summary_rows.append(["Growth %", round(growth_pct, 2)])

if profit_col != "None" and profit_col in df.columns:
    summary_rows.append(["Total Profit", round(df[profit_col].sum(skipna=True), 2)])

if cost_col != "None" and cost_col in df.columns:
    summary_rows.append(["Total Cost", round(df[cost_col].sum(skipna=True), 2)])

if quantity_col != "None" and quantity_col in df.columns:
    summary_rows.append(["Total Quantity", round(df[quantity_col].sum(skipna=True), 2)])

if margin_available:
    summary_rows.append(["Average Margin %", round(df["Computed_Margin_Percent"].mean(skipna=True), 2)])

summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

download_df = df.copy()
if time_col in download_df.columns:
    try:
        download_df[time_col] = pd.to_datetime(download_df[time_col], errors="coerce").dt.strftime("%Y-%m-%d")
    except Exception:
        pass

d1, d2 = st.columns(2)

with d1:
    st.download_button(
        "📥 Download Cleaned Data",
        data=download_df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_business_data.csv",
        mime="text/csv"
    )

with d2:
    st.download_button(
        "📥 Download Summary Report",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="business_summary_report.csv",
        mime="text/csv"
    )

card_close()

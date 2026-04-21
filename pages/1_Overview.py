import streamlit as st
import pandas as pd

from utils import (
    inject_css, section_title, card_open, card_close,
    load_csv, load_excel, load_demo_data, prepare_state,
    clean_numeric, parse_date, forecast_next_value, moving_average_forecast,
    fmt_num, safe_growth_percent, generate_auto_summary, recommendation_text,
    detect_anomalies_iqr
)

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")
inject_css()

st.markdown("""
<div class="hero-card">
    <div class="hero-badge">Overview</div>
    <div class="hero-title">📊 Upload and Explore</div>
    <div class="hero-subtitle">
        Load a dataset, configure your core metric, and generate decision-ready insights.
    </div>
    <div class="hero-muted">
        Supports sales, finance, and customer review datasets.
    </div>
</div>
""", unsafe_allow_html=True)

section_title("📊 Overview", "Upload data, configure metrics, and review KPIs.")

data_source = st.radio("Choose data source", ["Upload file", "Use demo dataset"], horizontal=True)

df_raw = None
if data_source == "Upload file":
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".csv"):
            df_raw = load_csv(uploaded_file)
        else:
            df_raw = load_excel(uploaded_file)
else:
    df_raw = load_demo_data()

if df_raw is None:
    st.stop()

state = prepare_state(df_raw)
df_raw = state["df_raw"]

st.session_state["dashboard_df_raw"] = df_raw
st.session_state["dashboard_text_col"] = state["text_col"]
st.session_state["dashboard_title_col"] = state["title_col"]

card_open()
section_title("📋 Dataset Health Check", "Quick overview of the uploaded data quality.")
h1, h2, h3, h4, h5 = st.columns(5)
h1.metric("Rows", len(df_raw))
h2.metric("Columns", len(df_raw.columns))
h3.metric("Missing Values", int(df_raw.isna().sum().sum()))
h4.metric("Duplicate Rows", int(df_raw.duplicated().sum()))
h5.metric("Numeric Columns", len(df_raw.select_dtypes(include="number").columns))
with st.expander("Preview Raw Data"):
    st.dataframe(df_raw.head(20), use_container_width=True)
card_close()

numeric_candidates = state["numeric_candidates"]
if not numeric_candidates:
    st.error("No numeric business columns were found in this dataset.")
    st.stop()

suggested_metric = state["suggested_metric"]
suggested_profit = state["suggested_profit"]
suggested_cost = state["suggested_cost"]
suggested_quantity = state["suggested_quantity"]
category_candidates = state["category_candidates"]

st.sidebar.markdown("### Configuration")
metric_col = st.sidebar.selectbox(
    "Main metric",
    numeric_candidates,
    index=numeric_candidates.index(suggested_metric) if suggested_metric in numeric_candidates else 0
)

date_options = ["generated_date"] + [col for col in df_raw.columns if col != "generated_date"]
date_col = st.sidebar.selectbox("Date column", date_options, index=0)

other_options = ["None"] + df_raw.columns.tolist()
profit_col = st.sidebar.selectbox("Profit column (optional)", other_options, index=other_options.index(suggested_profit) if suggested_profit in other_options else 0)
cost_col = st.sidebar.selectbox("Cost/Expense column (optional)", other_options, index=other_options.index(suggested_cost) if suggested_cost in other_options else 0)
quantity_col = st.sidebar.selectbox("Quantity column (optional)", other_options, index=other_options.index(suggested_quantity) if suggested_quantity in other_options else 0)

category_options = ["None"] + sorted(list(dict.fromkeys(category_candidates)))
category_col = st.sidebar.selectbox("Primary category", category_options, index=1 if len(category_options) > 1 else 0)

secondary_candidates = ["None"] + [c for c in category_candidates if c != category_col]
second_category_col = st.sidebar.selectbox("Secondary category", secondary_candidates)

df = df_raw.copy()
df = clean_numeric(df, metric_col)
df = df.dropna(subset=[metric_col])

has_real_date = False
time_col = None

df = parse_date(df, date_col)
if df[date_col].notna().sum() > 0:
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    has_real_date = True
    time_col = date_col
else:
    df["generated_date"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")
    time_col = "generated_date"

for col in [profit_col, cost_col, quantity_col]:
    if col != "None" and col in df.columns:
        df = clean_numeric(df, col)

margin_available = False
if profit_col != "None" and profit_col in df.columns:
    nonzero_metric = df[metric_col].replace(0, pd.NA)
    df["Computed_Margin_Percent"] = (df[profit_col] / nonzero_metric) * 100
    margin_available = True

prediction_lr, slope = forecast_next_value(df[metric_col])
prediction_ma = moving_average_forecast(df[metric_col], window=5)
prediction = prediction_lr

total_metric = df[metric_col].sum()
avg_metric = df[metric_col].mean()
max_metric = df[metric_col].max()
latest_metric = df[metric_col].iloc[-1]
growth_pct = safe_growth_percent(df[metric_col])

st.session_state["dashboard_df"] = df
st.session_state["dashboard_metric_col"] = metric_col
st.session_state["dashboard_time_col"] = time_col
st.session_state["dashboard_category_col"] = category_col
st.session_state["dashboard_second_category_col"] = second_category_col
st.session_state["dashboard_profit_col"] = profit_col
st.session_state["dashboard_cost_col"] = cost_col
st.session_state["dashboard_quantity_col"] = quantity_col
st.session_state["dashboard_margin_available"] = margin_available
st.session_state["dashboard_prediction"] = prediction
st.session_state["dashboard_prediction_lr"] = prediction_lr
st.session_state["dashboard_prediction_ma"] = prediction_ma
st.session_state["dashboard_slope"] = slope
st.session_state["dashboard_total_metric"] = total_metric
st.session_state["dashboard_avg_metric"] = avg_metric
st.session_state["dashboard_max_metric"] = max_metric
st.session_state["dashboard_latest_metric"] = latest_metric
st.session_state["dashboard_growth_pct"] = growth_pct

card_open()
section_title("📊 Executive Summary", "High-level KPI view of the selected metric.")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"Total {metric_col}", fmt_num(total_metric))
c2.metric(f"Average {metric_col}", fmt_num(avg_metric))
c3.metric(f"Highest {metric_col}", fmt_num(max_metric))
c4.metric(f"Predicted Next {metric_col}", fmt_num(prediction) if prediction is not None else "N/A")
c5.metric("Growth %", f"{growth_pct:.2f}%" if growth_pct is not None else "N/A")
card_close()

card_open()
section_title("📝 Automatic Summary", "Automatically generated insights based on your selected data.")
for line in generate_auto_summary(df, metric_col, category_col):
    st.markdown(f"<div class='info-card'>{line}</div>", unsafe_allow_html=True)
card_close()

card_open()
section_title("💡 Recommendations", "Action-oriented suggestions based on detected patterns.")
recommendations = recommendation_text(latest_metric, avg_metric, slope, metric_col)
if growth_pct is not None and slope is not None:
    if slope > 0 and growth_pct > 0:
        recommendations.append("Strong upward trend confirmed by both recent growth and forecast direction.")
    elif slope < 0 and growth_pct < 0:
        recommendations.append("Consistent downward trend detected in both recent growth and forecast direction.")
    else:
        recommendations.append("Mixed signals detected: recent growth and forecast trend are not fully aligned.")
for rec in recommendations:
    st.markdown(f"<div class='info-card'>{rec}</div>", unsafe_allow_html=True)
card_close()

card_open()
section_title("🚨 Anomaly Detection", "Detect unusual records using the IQR method.")
anomaly_df = detect_anomalies_iqr(df, metric_col)
anomaly_count = int(anomaly_df["is_anomaly"].sum())
anomaly_pct = (anomaly_count / len(anomaly_df)) * 100 if len(anomaly_df) > 0 else 0
st.session_state["dashboard_anomaly_df"] = anomaly_df
st.session_state["dashboard_anomaly_pct"] = anomaly_pct

if anomaly_count > 0:
    st.warning(f"{anomaly_count} unusual records were detected in {metric_col} ({anomaly_pct:.2f}% of data).")
    st.dataframe(anomaly_df[anomaly_df["is_anomaly"]].head(20), use_container_width=True)
else:
    st.success("No major anomalies detected.")
card_close()

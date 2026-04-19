import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Universal AI Business Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


@st.cache_data
def load_demo_data():
    np.random.seed(42)
    n = 180
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")

    item_types = np.random.choice(
        ["Dairy", "Fruits", "Meat", "Snacks", "Soft Drinks", "Baking Goods"],
        size=n
    )
    outlet_types = np.random.choice(
        ["Supermarket Type1", "Supermarket Type2", "Grocery Store"],
        size=n
    )
    outlet_sizes = np.random.choice(
        ["Small", "Medium", "High"],
        size=n
    )
    regions = np.random.choice(
        ["North", "South", "East", "West"],
        size=n
    )

    quantity = np.random.randint(5, 60, size=n)
    unit_price = np.round(np.random.uniform(8, 40, size=n), 2)
    cost = np.round(unit_price * np.random.uniform(0.5, 0.85, size=n), 2)
    revenue = np.round(quantity * unit_price + np.random.normal(0, 20, size=n), 2)
    profit = np.round(revenue - (quantity * cost), 2)
    visibility = np.round(np.random.uniform(0.01, 0.25, size=n), 3)

    return pd.DataFrame({
        "date": dates,
        "revenue": revenue,
        "profit": profit,
        "cost": cost,
        "quantity": quantity,
        "Item_Type": item_types,
        "Outlet_Type": outlet_types,
        "Outlet_Size": outlet_sizes,
        "Region": regions,
        "Item_Visibility": visibility
    })


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Common dataset-specific mappings
    mappings = {
        "Item_Outlet_Sales": "sales",
        "Sales": "sales",
        "Revenue": "revenue",
        "Profit": "profit",
        "Cost": "cost",
        "Quantity": "quantity",
        "Date": "date"
    }

    for old_col, new_col in mappings.items():
        if old_col in df.columns and new_col not in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    return df


def detect_date_column(df: pd.DataFrame):
    if "date" in df.columns:
        return "date"

    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower() or "day" in col.lower():
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > 0:
                return col
    return None


def detect_metric_candidates(df: pd.DataFrame):
    all_cols = df.columns.tolist()
    numeric_candidates = df.select_dtypes(include=np.number).columns.tolist()

    for col in all_cols:
        if col not in numeric_candidates:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                numeric_candidates.append(col)

    return list(dict.fromkeys(numeric_candidates))


def suggest_main_metric(columns):
    priorities = [
        "sales", "revenue", "profit", "amount", "income",
        "total", "value", "quantity", "cost"
    ]
    lower_map = {c.lower(): c for c in columns}

    for p in priorities:
        if p in lower_map:
            return lower_map[p]

    for c in columns:
        if any(word in c.lower() for word in priorities):
            return c

    return columns[0] if columns else None


def suggest_profit_column(columns):
    for c in columns:
        if "profit" in c.lower():
            return c
    return None


def suggest_cost_column(columns):
    for c in columns:
        if "cost" in c.lower() or "expense" in c.lower():
            return c
    return None


def suggest_quantity_column(columns):
    for c in columns:
        if "quantity" in c.lower() or "qty" in c.lower() or "units" in c.lower():
            return c
    return None


def suggest_category_columns(df, exclude_cols):
    candidates = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
            unique_count = df[col].nunique(dropna=True)
            if 2 <= unique_count <= 40:
                candidates.append(col)
    return candidates


def clean_numeric(df, col):
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_date(df, col):
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def forecast_next_value(series):
    series = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    if len(series) < 2:
        return None, None

    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values

    model = LinearRegression()
    model.fit(X, y)

    prediction = float(model.predict(np.array([[len(series)]]))[0])
    slope = float(model.coef_[0])
    return prediction, slope


def fmt_num(value):
    try:
        return f"{value:,.2f}"
    except Exception:
        return str(value)


def recommendation_text(latest_val, avg_val, slope, metric_name):
    recs = []

    if latest_val < avg_val:
        recs.append(f"The latest {metric_name} is below average. Review recent performance and possible causes.")
    else:
        recs.append(f"The latest {metric_name} is above average. Current performance looks healthy.")

    if slope is not None:
        if slope > 0:
            recs.append(f"The forecast trend for {metric_name} is positive.")
        elif slope < 0:
            recs.append(f"The forecast trend for {metric_name} is declining. Consider corrective actions.")
        else:
            recs.append(f"The forecast trend for {metric_name} is stable.")

    return recs


# -----------------------------
# Header
# -----------------------------
st.title("📊 Universal AI Business Dashboard")
st.markdown("### Analyze business CSV files with insights, filters, and forecasting")
st.caption("Built by Abdoulie J Bah")

st.markdown(
    """
Upload a CSV file and let the app automatically detect useful business columns.
You can also override the suggestions from the sidebar.
"""
)

# -----------------------------
# Data source
# -----------------------------
data_source = st.radio(
    "Choose data source",
    ["Upload CSV", "Use demo dataset"],
    horizontal=True
)

df_raw = None
uploaded_file = None

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df_raw = load_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")
            st.stop()
else:
    df_raw = load_demo_data()

if df_raw is None:
    st.stop()

df_raw = normalize_columns(df_raw)

# -----------------------------
# Health check
# -----------------------------
st.subheader("Dataset Health Check")

h1, h2, h3, h4, h5 = st.columns(5)
h1.metric("Rows", len(df_raw))
h2.metric("Columns", len(df_raw.columns))
h3.metric("Missing Values", int(df_raw.isna().sum().sum()))
h4.metric("Duplicate Rows", int(df_raw.duplicated().sum()))
h5.metric("Numeric Columns", len(df_raw.select_dtypes(include=np.number).columns))

with st.expander("Preview Raw Data"):
    st.dataframe(df_raw.head(20), use_container_width=True)

# -----------------------------
# Automatic suggestions
# -----------------------------
numeric_candidates = detect_metric_candidates(df_raw)
if not numeric_candidates:
    st.error("No numeric business columns were found in this dataset.")
    st.stop()

detected_date_col = detect_date_column(df_raw)
suggested_metric = suggest_main_metric(numeric_candidates)
suggested_profit = suggest_profit_column(df_raw.columns.tolist())
suggested_cost = suggest_cost_column(df_raw.columns.tolist())
suggested_quantity = suggest_quantity_column(df_raw.columns.tolist())

category_candidates = suggest_category_columns(
    df_raw,
    exclude_cols=[c for c in [suggested_metric, detected_date_col, suggested_profit, suggested_cost, suggested_quantity] if c]
)

# -----------------------------
# Sidebar config
# -----------------------------
st.sidebar.header("Configuration")

metric_col = st.sidebar.selectbox(
    "Main metric",
    numeric_candidates,
    index=numeric_candidates.index(suggested_metric) if suggested_metric in numeric_candidates else 0
)

date_options = ["None"] + df_raw.columns.tolist()
date_col = st.sidebar.selectbox(
    "Date column",
    date_options,
    index=date_options.index(detected_date_col) if detected_date_col in date_options else 0
)

profit_options = ["None"] + df_raw.columns.tolist()
profit_col = st.sidebar.selectbox(
    "Profit column (optional)",
    profit_options,
    index=profit_options.index(suggested_profit) if suggested_profit in profit_options else 0
)

cost_col = st.sidebar.selectbox(
    "Cost/Expense column (optional)",
    profit_options,
    index=profit_options.index(suggested_cost) if suggested_cost in profit_options else 0
)

quantity_col = st.sidebar.selectbox(
    "Quantity column (optional)",
    profit_options,
    index=profit_options.index(suggested_quantity) if suggested_quantity in profit_options else 0
)

category_options = ["None"] + category_candidates + [c for c in df_raw.columns if c not in category_candidates]
category_col = st.sidebar.selectbox("Primary category", category_options, index=1 if len(category_options) > 1 else 0)

second_category_options = ["None"] + [c for c in df_raw.columns if c != category_col]
second_category_col = st.sidebar.selectbox("Secondary category", second_category_options)

# -----------------------------
# Prepare working df
# -----------------------------
df = df_raw.copy()
df = clean_numeric(df, metric_col)
df = df.dropna(subset=[metric_col])

if df.empty:
    st.error("No valid values found in the selected main metric.")
    st.stop()

has_real_date = False
time_col = None

if date_col != "None":
    df = parse_date(df, date_col)
    if df[date_col].notna().sum() > 0:
        df = df.dropna(subset=[date_col]).sort_values(date_col)
        has_real_date = True
        time_col = date_col

if not has_real_date:
    df["Generated_Date"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")
    time_col = "Generated_Date"

# Optional numeric cleaning
optional_numeric_cols = [profit_col, cost_col, quantity_col]
for col in optional_numeric_cols:
    if col != "None" and col in df.columns:
        df = clean_numeric(df, col)

# Auto compute margin if possible
margin_available = False
if profit_col != "None" and metric_col:
    if profit_col in df.columns:
        df["Computed_Margin_Percent"] = (df[profit_col] / df[metric_col].replace(0, np.nan)) * 100
        margin_available = True

# -----------------------------
# Filters
# -----------------------------
st.sidebar.header("Filters")

if category_col != "None" and category_col in df.columns:
    vals = sorted(df[category_col].dropna().astype(str).unique().tolist())
    selected_vals = st.sidebar.multiselect(f"Filter {category_col}", vals, default=vals)
    if selected_vals:
        df = df[df[category_col].astype(str).isin(selected_vals)]

if second_category_col != "None" and second_category_col in df.columns:
    vals2 = sorted(df[second_category_col].dropna().astype(str).unique().tolist())
    selected_vals2 = st.sidebar.multiselect(f"Filter {second_category_col}", vals2, default=vals2)
    if selected_vals2:
        df = df[df[second_category_col].astype(str).isin(selected_vals2)]

metric_min = float(df[metric_col].min())
metric_max = float(df[metric_col].max())
if metric_min != metric_max:
    selected_range = st.sidebar.slider(
        f"{metric_col} range",
        min_value=metric_min,
        max_value=metric_max,
        value=(metric_min, metric_max)
    )
    df = df[df[metric_col].between(selected_range[0], selected_range[1])]

if df.empty:
    st.warning("No data remains after applying filters.")
    st.stop()

# -----------------------------
# Forecast
# -----------------------------
prediction, slope = forecast_next_value(df[metric_col])

# -----------------------------
# Executive summary cards
# -----------------------------
st.subheader("Executive Summary")

total_metric = df[metric_col].sum()
avg_metric = df[metric_col].mean()
max_metric = df[metric_col].max()
min_metric = df[metric_col].min()
latest_metric = df[metric_col].iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Total {metric_col}", fmt_num(total_metric))
c2.metric(f"Average {metric_col}", fmt_num(avg_metric))
c3.metric(f"Highest {metric_col}", fmt_num(max_metric))
c4.metric(f"Predicted Next {metric_col}", fmt_num(prediction) if prediction is not None else "N/A")

# Profit / margin row
extra_cols = st.columns(4)
if profit_col != "None" and profit_col in df.columns:
    total_profit = df[profit_col].sum(skipna=True)
    extra_cols[0].metric("Total Profit", fmt_num(total_profit))

if cost_col != "None" and cost_col in df.columns:
    total_cost = df[cost_col].sum(skipna=True)
    extra_cols[1].metric("Total Cost", fmt_num(total_cost))

if quantity_col != "None" and quantity_col in df.columns:
    total_qty = df[quantity_col].sum(skipna=True)
    extra_cols[2].metric("Total Quantity", fmt_num(total_qty))

if margin_available:
    avg_margin = df["Computed_Margin_Percent"].mean(skipna=True)
    extra_cols[3].metric("Average Margin %", fmt_num(avg_margin))

if not has_real_date:
    st.info("No reliable date column detected. Trend and forecast use generated dates for illustration.")

# -----------------------------
# Recommendations
# -----------------------------
st.subheader("Recommendations")

for rec in recommendation_text(latest_metric, avg_metric, slope, metric_col):
    st.write(f"- {rec}")

if category_col != "None" and category_col in df.columns:
    grouped = df.groupby(category_col)[metric_col].sum().sort_values(ascending=False)
    if len(grouped) > 0:
        st.write(f"- Best-performing {category_col}: **{grouped.index[0]}**")
        st.write(f"- Lowest-performing {category_col}: **{grouped.index[-1]}**")

if margin_available:
    low_margin_rows = df["Computed_Margin_Percent"].lt(df["Computed_Margin_Percent"].mean(skipna=True)).sum()
    st.write(f"- {low_margin_rows} rows are below the average margin. Review pricing or cost structure where relevant.")

# -----------------------------
# Main visuals
# -----------------------------
st.subheader("Visual Analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Trend",
    "Distribution",
    "Category",
    "Relationship",
    "Detailed Tables"
])

with tab1:
    trend_df = df[[time_col, metric_col]].copy().sort_values(time_col)

    line_chart = alt.Chart(trend_df).mark_line(point=True).encode(
        x=alt.X(f"{time_col}:T", title="Date"),
        y=alt.Y(f"{metric_col}:Q", title=metric_col),
        tooltip=[time_col, alt.Tooltip(f"{metric_col}:Q", format=".2f")]
    ).properties(height=360)

    st.altair_chart(line_chart, use_container_width=True)

with tab2:
    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{metric_col}:Q", bin=alt.Bin(maxbins=25), title=metric_col),
        y=alt.Y("count()", title="Count")
    ).properties(height=360)

    st.altair_chart(hist, use_container_width=True)

    if margin_available:
        st.markdown("#### Margin Distribution")
        margin_hist = alt.Chart(df.dropna(subset=["Computed_Margin_Percent"])).mark_bar().encode(
            x=alt.X("Computed_Margin_Percent:Q", bin=alt.Bin(maxbins=25), title="Margin %"),
            y=alt.Y("count()", title="Count")
        ).properties(height=320)
        st.altair_chart(margin_hist, use_container_width=True)

with tab3:
    if category_col != "None" and category_col in df.columns:
        grouped_df = (
            df.groupby(category_col)[metric_col]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        bar = alt.Chart(grouped_df).mark_bar().encode(
            x=alt.X(f"{metric_col}:Q", title=f"Total {metric_col}"),
            y=alt.Y(f"{category_col}:N", sort="-x", title=category_col),
            tooltip=[category_col, alt.Tooltip(f"{metric_col}:Q", format=".2f")]
        ).properties(height=360)

        st.altair_chart(bar, use_container_width=True)

        if second_category_col != "None" and second_category_col in df.columns:
            grouped_df2 = (
                df.groupby(second_category_col)[metric_col]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            st.markdown(f"#### {metric_col} by {second_category_col}")
            bar2 = alt.Chart(grouped_df2).mark_bar().encode(
                x=alt.X(f"{second_category_col}:N", title=second_category_col),
                y=alt.Y(f"{metric_col}:Q", title=f"Total {metric_col}"),
                tooltip=[second_category_col, alt.Tooltip(f"{metric_col}:Q", format=".2f")]
            ).properties(height=320)
            st.altair_chart(bar2, use_container_width=True)
    else:
        st.info("Choose a category column from the sidebar for grouped business analysis.")

with tab4:
    other_numeric = [c for c in detect_metric_candidates(df) if c != metric_col and c in df.columns]
    if other_numeric:
        x_col = st.selectbox("Select numeric X-axis", other_numeric)
        rel_df = df.copy()
        rel_df = clean_numeric(rel_df, x_col)
        rel_df = rel_df.dropna(subset=[x_col, metric_col])

        if not rel_df.empty:
            color_encoding = alt.Color(f"{category_col}:N") if category_col != "None" and category_col in rel_df.columns else alt.value("#1f77b4")

            scatter = alt.Chart(rel_df).mark_circle(size=80, opacity=0.65).encode(
                x=alt.X(f"{x_col}:Q", title=x_col),
                y=alt.Y(f"{metric_col}:Q", title=metric_col),
                color=color_encoding,
                tooltip=[x_col, alt.Tooltip(f"{metric_col}:Q", format=".2f")]
            ).properties(height=380)

            st.altair_chart(scatter, use_container_width=True)
        else:
            st.info("No valid rows available for relationship analysis.")
    else:
        st.info("No extra numeric columns available for relationship analysis.")

with tab5:
    left, right = st.columns(2)

    with left:
        st.markdown("#### Top 10 Rows by Main Metric")
        top_rows = df.sort_values(metric_col, ascending=False).head(10)
        st.dataframe(top_rows, use_container_width=True)

    with right:
        st.markdown("#### Bottom 10 Rows by Main Metric")
        bottom_rows = df.sort_values(metric_col, ascending=True).head(10)
        st.dataframe(bottom_rows, use_container_width=True)

    if category_col != "None" and category_col in df.columns:
        st.markdown("#### Category Summary")
        cat_summary = (
            df.groupby(category_col)[metric_col]
            .agg(["sum", "mean", "count"])
            .sort_values("sum", ascending=False)
            .reset_index()
        )
        cat_summary.columns = [category_col, f"Total {metric_col}", f"Average {metric_col}", "Records"]
        st.dataframe(cat_summary, use_container_width=True)

# -----------------------------
# Downloads
# -----------------------------
st.subheader("Downloads")

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

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "This dashboard is built for flexible business CSV analysis, interactive filtering, and simple machine-learning-based forecasting."
)

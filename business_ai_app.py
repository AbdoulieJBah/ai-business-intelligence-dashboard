import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="AI Business Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------
# Helper functions
# -------------------------
def load_demo_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 120
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
    fat_content = np.random.choice(["Low Fat", "Regular"], size=n)
    visibility = np.round(np.random.uniform(0.01, 0.25, size=n), 3)
    item_weight = np.round(np.random.uniform(5, 20, size=n), 2)
    mrp = np.round(np.random.uniform(50, 300, size=n), 2)

    base_sales = 200 + np.arange(n) * 1.2 + np.random.normal(0, 25, size=n)
    category_effect = pd.Series(item_types).map({
        "Dairy": 20, "Fruits": 10, "Meat": 30, "Snacks": 25,
        "Soft Drinks": 15, "Baking Goods": 12
    }).values
    outlet_effect = pd.Series(outlet_types).map({
        "Supermarket Type1": 35, "Supermarket Type2": 20, "Grocery Store": -10
    }).values

    sales = np.round(base_sales + category_effect + outlet_effect, 2)

    return pd.DataFrame({
        "date": dates,
        "sales": sales,
        "Item_Type": item_types,
        "Outlet_Type": outlet_types,
        "Outlet_Size": outlet_sizes,
        "Item_Fat_Content": fat_content,
        "Item_Visibility": visibility,
        "Item_Weight": item_weight,
        "Item_MRP": mrp
    })


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Big Mart support
    if "Item_Outlet_Sales" in df.columns and "sales" not in df.columns:
        df.rename(columns={"Item_Outlet_Sales": "sales"}, inplace=True)

    # Try to detect date-like column if "date" not already present
    if "date" not in df.columns:
        possible_date_cols = [c for c in df.columns if "date" in c.lower()]
        if possible_date_cols:
            chosen = possible_date_cols[0]
            df["date"] = pd.to_datetime(df[chosen], errors="coerce")

    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = df.copy()

    st.sidebar.header("Filters")

    # Categorical filters
    categorical_candidates = [
        "Item_Type", "Outlet_Type", "Outlet_Size",
        "Item_Fat_Content", "Outlet_Location_Type"
    ]

    for col in categorical_candidates:
        if col in filtered_df.columns:
            options = sorted(filtered_df[col].dropna().astype(str).unique().tolist())
            selected = st.sidebar.multiselect(f"{col}", options, default=options)
            if selected:
                filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]

    # Numeric filters
    numeric_candidates = ["sales", "Item_MRP", "Item_Visibility", "Item_Weight"]
    for col in numeric_candidates:
        if col in filtered_df.columns:
            numeric_series = pd.to_numeric(filtered_df[col], errors="coerce").dropna()
            if not numeric_series.empty:
                min_val = float(numeric_series.min())
                max_val = float(numeric_series.max())
                if min_val != max_val:
                    selected_range = st.sidebar.slider(
                        f"{col} range",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=(float(min_val), float(max_val))
                    )
                    filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce")
                    filtered_df = filtered_df[
                        filtered_df[col].between(selected_range[0], selected_range[1], inclusive="both")
                    ]

    return filtered_df


def business_summary(df: pd.DataFrame, target_col: str) -> dict:
    summary = {}
    summary["rows"] = len(df)
    summary["total"] = float(df[target_col].sum())
    summary["average"] = float(df[target_col].mean())
    summary["max"] = float(df[target_col].max())
    summary["min"] = float(df[target_col].min())
    return summary


def train_forecast_model(df: pd.DataFrame, target_col: str):
    model_df = df[[target_col]].dropna().reset_index(drop=True)
    if len(model_df) < 2:
        return None, None, None

    X = np.arange(len(model_df)).reshape(-1, 1)
    y = model_df[target_col].values

    model = LinearRegression()
    model.fit(X, y)

    next_index = np.array([[len(model_df)]])
    prediction = float(model.predict(next_index)[0])
    slope = float(model.coef_[0])

    return model, prediction, slope


def generate_exec_summary(df: pd.DataFrame, target_col: str) -> list[str]:
    insights = []

    avg_val = df[target_col].mean()
    latest_val = df[target_col].iloc[-1]

    if latest_val < avg_val:
        insights.append(f"The latest {target_col} value is below the overall average.")
    else:
        insights.append(f"The latest {target_col} value is above the overall average.")

    if len(df) >= 2:
        if df[target_col].iloc[-1] < df[target_col].iloc[-2]:
            insights.append(f"Recent {target_col} shows a short-term decline.")
        else:
            insights.append(f"Recent {target_col} looks stable or improving.")

    if "Item_Type" in df.columns:
        top_category = (
            df.groupby("Item_Type")[target_col]
            .sum()
            .sort_values(ascending=False)
            .head(1)
        )
        if not top_category.empty:
            insights.append(f"Top product category by {target_col}: {top_category.index[0]}.")

    if "Outlet_Type" in df.columns:
        top_outlet = (
            df.groupby("Outlet_Type")[target_col]
            .sum()
            .sort_values(ascending=False)
            .head(1)
        )
        if not top_outlet.empty:
            insights.append(f"Best outlet type by {target_col}: {top_outlet.index[0]}.")

    return insights


def make_downloadable_summary(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    rows = []

    rows.append(["Rows loaded", len(df)])
    rows.append([f"Total {target_col}", round(df[target_col].sum(), 2)])
    rows.append([f"Average {target_col}", round(df[target_col].mean(), 2)])
    rows.append([f"Highest {target_col}", round(df[target_col].max(), 2)])
    rows.append([f"Lowest {target_col}", round(df[target_col].min(), 2)])

    if "Item_Type" in df.columns:
        best_cat = df.groupby("Item_Type")[target_col].sum().sort_values(ascending=False).head(1)
        if not best_cat.empty:
            rows.append([f"Top category by {target_col}", best_cat.index[0]])

    if "Outlet_Type" in df.columns:
        best_outlet = df.groupby("Outlet_Type")[target_col].sum().sort_values(ascending=False).head(1)
        if not best_outlet.empty:
            rows.append([f"Top outlet type by {target_col}", best_outlet.index[0]])

    return pd.DataFrame(rows, columns=["Metric", "Value"])


# -------------------------
# Header
# -------------------------
st.title("📊 AI Business Intelligence Dashboard")
st.markdown("### Smart insights, interactive analysis, and machine learning forecasts")
st.caption("Built by Abdoulie Bah")

st.info(
    "Upload a CSV file or use demo mode. "
    "Best results come from datasets with a sales column (or Big Mart's Item_Outlet_Sales). "
    "If a real date column is missing, the app will still work, but forecast timing should be treated as demo-only."
)

# -------------------------
# Data source
# -------------------------
data_mode = st.radio(
    "Choose data source",
    ["Upload CSV", "Use demo dataset"],
    horizontal=True
)

uploaded_file = None
df_raw = None

if data_mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
else:
    df_raw = load_demo_data()

if df_raw is None:
    st.stop()

df = prepare_data(df_raw)

# -------------------------
# Dataset health
# -------------------------
st.subheader("Dataset Health Check")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Rows", len(df))
col2.metric("Columns", len(df.columns))
col3.metric("Missing Values", int(df.isna().sum().sum()))
col4.metric("Duplicate Rows", int(df.duplicated().sum()))
col5.metric("Numeric Columns", len(df.select_dtypes(include=np.number).columns))

with st.expander("Preview Raw Data"):
    st.dataframe(df.head(20), use_container_width=True)

# -------------------------
# Target column selection
# -------------------------
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

preferred_targets = []
if "sales" in df.columns:
    preferred_targets.append("sales")
for col in ["Item_MRP", "Item_Visibility", "Item_Weight"]:
    if col in numeric_columns and col not in preferred_targets:
        preferred_targets.append(col)

remaining_targets = [c for c in numeric_columns if c not in preferred_targets]
target_options = preferred_targets + remaining_targets

if not target_options:
    st.error("No numeric columns were found in this dataset.")
    st.stop()

default_target = "sales" if "sales" in target_options else target_options[0]
target_col = st.selectbox("Select target metric", target_options, index=target_options.index(default_target))

# Clean target column
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[target_col])

# Real date detection
has_real_date = False
if "date" in df.columns:
    parsed_date = pd.to_datetime(df["date"], errors="coerce")
    if parsed_date.notna().sum() > 0:
        df["date"] = parsed_date
        df = df.dropna(subset=["date"]).sort_values("date")
        has_real_date = True

# Add generated date only if missing
if "date" not in df.columns or not has_real_date:
    df["date"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")
    has_real_date = False

# -------------------------
# Apply filters
# -------------------------
df_filtered = apply_filters(df)

if df_filtered.empty:
    st.warning("No data remains after applying the selected filters.")
    st.stop()

# -------------------------
# Executive summary
# -------------------------
st.subheader("Executive Summary")

summary = business_summary(df_filtered, target_col)
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Total {target_col}", f"{summary['total']:.2f}")
c2.metric(f"Average {target_col}", f"{summary['average']:.2f}")
c3.metric(f"Highest {target_col}", f"{summary['max']:.2f}")
c4.metric(f"Lowest {target_col}", f"{summary['min']:.2f}")

for insight in generate_exec_summary(df_filtered, target_col):
    st.write(f"- {insight}")

if not has_real_date:
    st.warning(
        "This dataset does not contain a reliable real date column. "
        "The time-based chart and forecast are shown in demo mode using generated dates."
    )

# -------------------------
# Detailed data preview
# -------------------------
with st.expander("Preview Filtered Data"):
    st.dataframe(df_filtered.head(50), use_container_width=True)

# -------------------------
# Charts
# -------------------------
st.subheader("Performance Visuals")

tab1, tab2, tab3, tab4 = st.tabs([
    "Trend",
    "Category Analysis",
    "Distribution",
    "Relationship"
])

with tab1:
    st.markdown("#### Trend Over Time")
    trend_df = df_filtered[["date", target_col]].copy().sort_values("date")

    line_chart = alt.Chart(trend_df).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y(f"{target_col}:Q", title=target_col),
        tooltip=["date:T", alt.Tooltip(f"{target_col}:Q", format=".2f")]
    ).properties(height=350)

    st.altair_chart(line_chart, use_container_width=True)

with tab2:
    shown_any = False

    if "Item_Type" in df_filtered.columns:
        st.markdown(f"#### Top Product Categories by {target_col}")
        category_df = (
            df_filtered.groupby("Item_Type")[target_col]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        bar = alt.Chart(category_df).mark_bar().encode(
            x=alt.X(f"{target_col}:Q", title=target_col),
            y=alt.Y("Item_Type:N", sort="-x", title="Item Type"),
            tooltip=["Item_Type:N", alt.Tooltip(f"{target_col}:Q", format=".2f")]
        ).properties(height=350)
        st.altair_chart(bar, use_container_width=True)
        shown_any = True

    if "Outlet_Type" in df_filtered.columns:
        st.markdown(f"#### Sales by Outlet Type")
        outlet_df = (
            df_filtered.groupby("Outlet_Type")[target_col]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        bar2 = alt.Chart(outlet_df).mark_bar().encode(
            x=alt.X("Outlet_Type:N", title="Outlet Type"),
            y=alt.Y(f"{target_col}:Q", title=target_col),
            tooltip=["Outlet_Type:N", alt.Tooltip(f"{target_col}:Q", format=".2f")]
        ).properties(height=350)
        st.altair_chart(bar2, use_container_width=True)
        shown_any = True

    if "Outlet_Size" in df_filtered.columns:
        st.markdown(f"#### Average {target_col} by Outlet Size")
        size_df = (
            df_filtered.groupby("Outlet_Size")[target_col]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        bar3 = alt.Chart(size_df).mark_bar().encode(
            x=alt.X("Outlet_Size:N", title="Outlet Size"),
            y=alt.Y(f"{target_col}:Q", title=f"Average {target_col}"),
            tooltip=["Outlet_Size:N", alt.Tooltip(f"{target_col}:Q", format=".2f")]
        ).properties(height=300)
        st.altair_chart(bar3, use_container_width=True)
        shown_any = True

    if not shown_any:
        st.info("No supported category columns were found for this dataset.")

with tab3:
    st.markdown(f"#### Distribution of {target_col}")

    hist = alt.Chart(df_filtered).mark_bar().encode(
        x=alt.X(f"{target_col}:Q", bin=alt.Bin(maxbins=25), title=target_col),
        y=alt.Y("count()", title="Count"),
        tooltip=["count()"]
    ).properties(height=350)

    st.altair_chart(hist, use_container_width=True)

with tab4:
    relationship_candidates = ["Item_MRP", "Item_Visibility", "Item_Weight"]
    available_candidates = [c for c in relationship_candidates if c in df_filtered.columns and c != target_col]

    if available_candidates:
        x_col = st.selectbox("Choose X-axis variable", available_candidates)
        scatter = alt.Chart(df_filtered).mark_circle(size=70, opacity=0.6).encode(
            x=alt.X(f"{x_col}:Q", title=x_col),
            y=alt.Y(f"{target_col}:Q", title=target_col),
            color=alt.Color("Item_Type:N") if "Item_Type" in df_filtered.columns else alt.value("#1f77b4"),
            tooltip=[x_col, alt.Tooltip(f"{target_col}:Q", format=".2f")]
        ).properties(height=380)
        st.altair_chart(scatter, use_container_width=True)
    else:
        st.info("No suitable numeric columns were found for relationship analysis.")

# -------------------------
# Smarter insights
# -------------------------
st.subheader("Business Insights")

insight_cols = st.columns(2)

with insight_cols[0]:
    if "Item_Type" in df_filtered.columns:
        top_categories = (
            df_filtered.groupby("Item_Type")[target_col]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        bottom_categories = (
            df_filtered.groupby("Item_Type")[target_col]
            .sum()
            .sort_values(ascending=True)
            .head(5)
        )

        st.markdown(f"#### Top 5 Categories by {target_col}")
        st.dataframe(top_categories.reset_index().rename(columns={target_col: f"Total {target_col}"}), use_container_width=True)

        st.markdown(f"#### Bottom 5 Categories by {target_col}")
        st.dataframe(bottom_categories.reset_index().rename(columns={target_col: f"Total {target_col}"}), use_container_width=True)

with insight_cols[1]:
    if "Outlet_Type" in df_filtered.columns:
        outlet_perf = (
            df_filtered.groupby("Outlet_Type")[target_col]
            .agg(["sum", "mean", "count"])
            .sort_values("sum", ascending=False)
            .reset_index()
        )
        outlet_perf.columns = ["Outlet Type", f"Total {target_col}", f"Average {target_col}", "Records"]
        st.markdown("#### Outlet Performance Summary")
        st.dataframe(outlet_perf, use_container_width=True)

# -------------------------
# Forecast
# -------------------------
st.subheader("Machine Learning Forecast")

model, prediction, slope = train_forecast_model(df_filtered, target_col)

if prediction is None:
    st.info("At least 2 rows are needed to generate a forecast.")
else:
    m1, m2 = st.columns(2)
    with m1:
        st.metric(f"Predicted Next {target_col}", f"{prediction:.2f}")
    with m2:
        if slope > 0:
            st.success("The model detects an upward trend.")
        elif slope < 0:
            st.warning("The model detects a downward trend.")
        else:
            st.info("The model detects a flat trend.")

# -------------------------
# Downloads
# -------------------------
st.subheader("Downloads")

download_summary = make_downloadable_summary(df_filtered, target_col)

download_df = df_filtered.copy()
download_df["date"] = pd.to_datetime(download_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

d1, d2 = st.columns(2)

with d1:
    st.download_button(
        label="📥 Download Cleaned Data",
        data=download_df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_business_data.csv",
        mime="text/csv"
    )

with d2:
    st.download_button(
        label="📥 Download Summary Report",
        data=download_summary.to_csv(index=False).encode("utf-8"),
        file_name="business_summary_report.csv",
        mime="text/csv"
    )

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption(
    "This dashboard is designed for business data exploration, trend analysis, and simple ML forecasting. "
    "For datasets without a real date field, time-based outputs should be treated as illustrative."
)

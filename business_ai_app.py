import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Universal AI Business Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

@st.cache_data
def load_excel(file):
    return pd.read_excel(file)

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
        name = col.lower()
        if "date" in name or "time" in name or "day" in name:
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
    preferred = []
    fallback = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        unique_count = df[col].nunique(dropna=True)

        # Accept object columns OR low unique numeric columns
        if df[col].dtype == "object" or unique_count < 50:

            col_lower = col.lower()

            if any(k in col_lower for k in ["category", "type", "region", "segment", "product", "item", "outlet", "store"]):
                preferred.append(col)
            else:
                fallback.append(col)

    # 🔥 fallback: if still empty, force include some columns
    if len(preferred + fallback) == 0:
        fallback = df.columns.tolist()

    return preferred + fallback

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

def generate_auto_summary(df, metric_col, category_col="None"):
    summary = []

    total = df[metric_col].sum()
    avg = df[metric_col].mean()
    mx = df[metric_col].max()
    mn = df[metric_col].min()

    summary.append(f"Total {metric_col} is {total:,.2f}.")
    summary.append(f"Average {metric_col} is {avg:,.2f}.")
    summary.append(f"Highest {metric_col} is {mx:,.2f}, while the lowest is {mn:,.2f}.")

    if len(df) >= 2:
        first = df[metric_col].iloc[0]
        last = df[metric_col].iloc[-1]
        if first != 0:
            growth = ((last - first) / abs(first)) * 100
            if growth > 0:
                summary.append(f"The metric increased by {growth:.2f}% over the selected period.")
            elif growth < 0:
                summary.append(f"The metric decreased by {abs(growth):.2f}% over the selected period.")
            else:
                summary.append("The metric remained stable over the selected period.")

    if category_col != "None" and category_col in df.columns:
        grouped = df.groupby(category_col)[metric_col].sum().sort_values(ascending=False)
        if len(grouped) > 0:
            summary.append(f"The best-performing {category_col} is {grouped.index[0]}.")
            summary.append(f"The lowest-performing {category_col} is {grouped.index[-1]}.")

    return summary


def safe_growth_percent(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < 2:
        return None
    first = series.iloc[0]
    last = series.iloc[-1]
    if first == 0:
        return None
    return ((last - first) / abs(first)) * 100
    
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


basic_stopwords = {
    "the", "and", "for", "that", "with", "this", "was", "are", "but", "have",
    "not", "you", "all", "from", "they", "your", "has", "had", "his", "her",
    "its", "our", "out", "too", "very", "can", "will", "would", "could", "should",
    "than", "then", "when", "what", "where", "which", "while", "were", "been",
    "also", "there", "their", "them", "into", "about", "after", "before", "because",
    "just", "more", "most", "some", "such", "only", "over", "much", "many", "any",
    "amazon", "product", "products", "item", "items", "buy", "bought", "use", "used",
    "one", "two", "get", "got", "good", "great"
}


def get_word_counts(text_series, stopwords=None, top_n=15):
    if stopwords is None:
        stopwords = set()

    words = []
    for text in text_series.dropna():
        cleaned = clean_text(text)
        words.extend([w for w in cleaned.split() if w not in stopwords and len(w) > 2])

    return Counter(words).most_common(top_n)


def simple_sentiment_from_rating(rating):
    try:
        rating = float(rating)
        if rating >= 4:
            return "Positive"
        elif rating >= 3:
            return "Neutral"
        else:
            return "Negative"
    except Exception:
        return "Unknown"


@st.cache_data
def build_tfidf_keywords(text_series, top_n=20):
    cleaned_docs = [clean_text(x) for x in text_series.fillna("").astype(str)]
    cleaned_docs = [doc for doc in cleaned_docs if doc.strip()]

    if len(cleaned_docs) < 3:
        return pd.DataFrame(columns=["Keyword", "Score"]), None, None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2
    )
    X = vectorizer.fit_transform(cleaned_docs)

    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())

    top_idx = np.argsort(scores)[::-1][:top_n]
    keyword_df = pd.DataFrame({
        "Keyword": terms[top_idx],
        "Score": scores[top_idx]
    })

    return keyword_df, X, vectorizer


@st.cache_data
def cluster_reviews(text_series, n_clusters=3):
    cleaned_docs = [clean_text(x) for x in text_series.fillna("").astype(str)]
    cleaned_docs = [doc for doc in cleaned_docs if doc.strip()]

    if len(cleaned_docs) < max(10, n_clusters):
        return None, None, None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2
    )
    X = vectorizer.fit_transform(cleaned_docs)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    terms = vectorizer.get_feature_names_out()
    top_terms_per_cluster = {}

    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        top_indices = center.argsort()[::-1][:8]
        top_terms_per_cluster[i] = [terms[idx] for idx in top_indices]

    clustered_df = pd.DataFrame({
        "clean_text": cleaned_docs,
        "cluster": labels
    })
    return clustered_df, top_terms_per_cluster, vectorizer
@st.cache_data
def detect_anomalies_iqr(df, metric_col):
    temp = df.copy()
    temp[metric_col] = pd.to_numeric(temp[metric_col], errors="coerce")
    temp = temp.dropna(subset=[metric_col])

    if len(temp) < 5:
        temp["is_anomaly"] = False
        return temp

    q1 = temp[metric_col].quantile(0.25)
    q3 = temp[metric_col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    temp["is_anomaly"] = (temp[metric_col] < lower) | (temp[metric_col] > upper)
    return temp
@st.cache_data    
def moving_average_forecast(series, window=5):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < window:
        return None
    return float(series.tail(window).mean())

# -----------------------------
# Header
# -----------------------------
st.title("📊 Universal AI Business Dashboard")
st.markdown("### Analyze business files with insights, filters, and forecasting")
st.caption("Built by Abdoulie J Bah")

st.write(
    "Upload a CSV or Excel file and let the app automatically detect useful business columns. "
    "You can override the suggestions from the sidebar."
)

with st.expander("How to use this dashboard"):
    st.write(
        """
        1. Upload a CSV or Excel file, or use the demo dataset.
        2. Select the main metric, date column, and optional business columns from the sidebar.
        3. Use filters to narrow the analysis.
        4. Review the executive summary, recommendations, visuals, and downloads.
        """
    )

# -----------------------------
# Data source
# -----------------------------
data_source = st.radio(
    "Choose data source",
    ["Upload file", "Use demo dataset"],
    horizontal=True
)

df_raw = None
uploaded_file = None

if data_source == "Upload file":
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df_raw = load_csv(uploaded_file)
            else:
                df_raw = load_excel(uploaded_file)

            st.success("File uploaded successfully!")

        except Exception as e:
            st.error(f"Could not read the file: {e}")
            st.stop()
else:
    df_raw = load_demo_data()

if df_raw is None:
    st.stop()

df_raw = normalize_columns(df_raw)

df_raw["generated_date"] = pd.date_range(start="2024-01-01", periods=len(df_raw), freq="D")
time_col = "generated_date"
    

# -----------------------------
# NLP preparation
# -----------------------------
text_col = None
title_col = None

if "review_content" in df_raw.columns:
    text_col = "review_content"
elif "about_product" in df_raw.columns:
    text_col = "about_product"

if "review_title" in df_raw.columns:
    title_col = "review_title"

# --- Clean text
if text_col is not None:
    df_raw["clean_text"] = df_raw[text_col].fillna("").astype(str).apply(clean_text)
    df_raw["review_length"] = df_raw["clean_text"].apply(lambda x: len(x.split()))

# --- FIX: convert rating to numeric FIRST
if "rating" in df_raw.columns:
    df_raw["rating"] = pd.to_numeric(df_raw["rating"], errors="coerce")

    df_raw["sentiment_label"] = df_raw["rating"].apply(simple_sentiment_from_rating)

# -----------------------------
# Health check
# -----------------------------
st.subheader("📋 Dataset Health Check")

h1, h2, h3, h4, h5 = st.columns(5)
h1.metric("Rows", len(df_raw))
h2.metric("Columns", len(df_raw.columns))
h3.metric("Missing Values", int(df_raw.isna().sum().sum()))
h4.metric("Duplicate Rows", int(df_raw.duplicated().sum()))
h5.metric("Numeric Columns", len(df_raw.select_dtypes(include=np.number).columns))

with st.expander("Preview Raw Data"):
    st.dataframe(df_raw.head(20), use_container_width=True)

st.divider()

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

st.subheader("🧠 Business Insight")

if growth_pct is not None:
    if growth_pct > 0:
        st.success("Business performance is improving over time.")
    else:
        st.error("Business performance is declining and needs attention.")

if anomaly_pct > 5:
    st.warning("High anomaly rate detected — investigate data quality or unusual events.")

# -----------------------------
# Sidebar config
# -----------------------------
st.sidebar.header("Configuration")

metric_col = st.sidebar.selectbox(
    "Main metric",
    numeric_candidates,
    index=numeric_candidates.index(suggested_metric) if suggested_metric in numeric_candidates else 0
)


date_options = ["generated_date"] + [
    col for col in df_raw.columns if col != "generated_date"
]

default_index = 0

if detected_date_col and detected_date_col in date_options:
    default_index = date_options.index(detected_date_col)

date_col = st.sidebar.selectbox(
    "Date column",
    date_options,
    index=default_index
)

other_options = ["None"] + df_raw.columns.tolist()

profit_col = st.sidebar.selectbox(
    "Profit column (optional)",
    other_options,
    index=other_options.index(suggested_profit) if suggested_profit in other_options else 0
)

cost_col = st.sidebar.selectbox(
    "Cost/Expense column (optional)",
    other_options,
    index=other_options.index(suggested_cost) if suggested_cost in other_options else 0
)

quantity_col = st.sidebar.selectbox(
    "Quantity column (optional)",
    other_options,
    index=other_options.index(suggested_quantity) if suggested_quantity in other_options else 0
)

category_options = ["None"] + sorted(list(dict.fromkeys(category_candidates)))
category_col = st.sidebar.selectbox(
    "Primary category",
    category_options,
    index=1 if len(category_options) > 1 else 0
)

secondary_candidates = ["None"] + [c for c in category_candidates if c != category_col]
second_category_col = st.sidebar.selectbox("Secondary category", secondary_candidates)

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

for col in [profit_col, cost_col, quantity_col]:
    if col != "None" and col in df.columns:
        df = clean_numeric(df, col)

margin_available = False
if profit_col != "None" and profit_col in df.columns:
    nonzero_metric = df[metric_col].replace(0, np.nan)
    df["Computed_Margin_Percent"] = (df[profit_col] / nonzero_metric) * 100
    margin_available = True

# -----------------------------
# Filters
# -----------------------------
st.sidebar.header("Filters")

if has_real_date:
    min_date = df[time_col].min().date()
    max_date = df[time_col].max().date()

    selected_dates = st.sidebar.date_input(
        "Filter date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
        df = df[(df[time_col].dt.date >= start_date) & (df[time_col].dt.date <= end_date)]
    elif not isinstance(selected_dates, tuple):
        df = df[df[time_col].dt.date == selected_dates]

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
# Forecast and summary values
# -----------------------------
prediction_lr, slope = forecast_next_value(df[metric_col])
prediction_ma = moving_average_forecast(df[metric_col], window=5)
prediction = prediction_lr
total_metric = df[metric_col].sum()
avg_metric = df[metric_col].mean()
max_metric = df[metric_col].max()
min_metric = df[metric_col].min()
latest_metric = df[metric_col].iloc[-1]
growth_pct = safe_growth_percent(df[metric_col])

st.subheader("🔮 Forecasting")

f1, f2 = st.columns(2)
f1.metric("Linear Forecast", fmt_num(prediction_lr) if prediction_lr is not None else "N/A")
f2.metric("Moving Average Forecast", fmt_num(prediction_ma) if prediction_ma is not None else "N/A")

st.caption("Linear forecast uses a trend line. Moving average forecast uses the recent 5 records.")

# -----------------------------
# Forecast calculation
# -----------------------------
prediction, slope = forecast_next_value(df[metric_col])

# -----------------------------
# Executive summary
# -----------------------------
st.subheader("📊 Executive Summary")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"Total {metric_col}", fmt_num(total_metric))
c2.metric(f"Average {metric_col}", fmt_num(avg_metric))
c3.metric(f"Highest {metric_col}", fmt_num(max_metric))
c4.metric(f"Predicted Next {metric_col}", fmt_num(prediction) if prediction is not None else "N/A")
c5.metric("Growth %", f"{growth_pct:.2f}%" if growth_pct is not None else "N/A")

extra_cols = st.columns(4)

if profit_col != "None" and profit_col in df.columns:
    extra_cols[0].metric("Total Profit", fmt_num(df[profit_col].sum(skipna=True)))

if cost_col != "None" and cost_col in df.columns:
    extra_cols[1].metric("Total Cost", fmt_num(df[cost_col].sum(skipna=True)))

if quantity_col != "None" and quantity_col in df.columns:
    extra_cols[2].metric("Total Quantity", fmt_num(df[quantity_col].sum(skipna=True)))

if margin_available:
    extra_cols[3].metric("Average Margin %", fmt_num(df["Computed_Margin_Percent"].mean(skipna=True)))

if not has_real_date:
    st.info("No reliable date column detected. Trend and forecast use generated dates for illustration.")

st.divider()

st.subheader("📝 Automatic Summary")
st.caption("Automatically generated insights based on your selected data.")
for line in generate_auto_summary(df, metric_col, category_col):
    st.markdown(f"• {line}")

# -----------------------------
# Recommendations
# -----------------------------
st.subheader("💡 Recommendations")

recommendations = recommendation_text(latest_metric, avg_metric, slope, metric_col)

if growth_pct is not None:
    if slope is not None:
        if slope > 0 and growth_pct > 0:
            recommendations.append("Strong upward trend confirmed by both recent growth and forecast direction.")
        elif slope < 0 and growth_pct < 0:
            recommendations.append("Consistent downward trend detected in both recent growth and forecast direction.")
        else:
            recommendations.append("Mixed signals detected: recent growth and forecast trend are not fully aligned.")

for rec in recommendations:
    st.write(f"- {rec}")

if category_col != "None" and category_col in df.columns:
    grouped = df.groupby(category_col)[metric_col].sum().sort_values(ascending=False)
    if len(grouped) > 0:
        st.write(f"- Best-performing {category_col}: **{grouped.index[0]}**")
        st.write(f"- Lowest-performing {category_col}: **{grouped.index[-1]}**")

if margin_available:
    low_margin_rows = df["Computed_Margin_Percent"].lt(
        df["Computed_Margin_Percent"].mean(skipna=True)
    ).sum()
    st.write(
        f"- {low_margin_rows} rows are below the average margin. Review pricing or cost structure where relevant."
    )

if prediction is not None:
    st.caption("Forecast based on a simple linear regression trend model.")

st.divider()
# -----------------------------
# AI Insights
# -----------------------------
st.subheader("🤖 AI Insights")

insights = []

# --- Trend insight
if slope is not None:
    if slope > 0:
        insights.append(f"{metric_col} is showing a consistent upward trend, indicating business growth.")
    elif slope < 0:
        insights.append(f"{metric_col} is declining over time, which may require immediate attention.")
    else:
        insights.append(f"{metric_col} trend is stable with no significant change.")

# --- Growth insight
if growth_pct is not None:
    if growth_pct > 10:
        insights.append(f"Strong growth detected (+{growth_pct:.2f}%). The business is expanding well.")
    elif growth_pct < -10:
        insights.append(f"Significant decline detected ({growth_pct:.2f}%). Investigate potential issues.")
    else:
        insights.append(f"Growth is relatively stable ({growth_pct:.2f}%).")

# --- Volatility insight
std_dev = df[metric_col].std()
if std_dev > avg_metric * 0.5:
    insights.append("High variability detected in performance, indicating inconsistent results.")
else:
    insights.append("Performance is relatively stable with low variability.")

# --- Category insight
if category_col != "None" and category_col in df.columns:
    if df[category_col].dtype == "object":
        grouped = df.groupby(category_col)[metric_col].sum().sort_values(ascending=False)

        if len(grouped) > 1:
            best = grouped.index[0]
            worst = grouped.index[-1]

            insights.append(f"{best} is the top-performing {category_col}, while {worst} is underperforming.")

# --- Profit insight
if profit_col != "None" and profit_col in df.columns:
    total_profit = df[profit_col].sum()
    if total_profit > 0:
        insights.append("The business is overall profitable.")
    else:
        insights.append("The business is operating at a loss.")

# --- Display insights
for i, insight in enumerate(insights):
    st.write(f"🔹 {insight}")
st.success("These insights are automatically generated using statistical analysis and trend detection.")

st.subheader("🚨 Anomaly Detection")

anomaly_df = detect_anomalies_iqr(df, metric_col)
anomaly_count = int(anomaly_df["is_anomaly"].sum())

if anomaly_count > 0:
    st.warning(f"{anomaly_count} unusual records were detected in {metric_col}.")
    st.dataframe(anomaly_df[anomaly_df["is_anomaly"]].head(20), use_container_width=True)
else:
    st.success("No major anomalies detected.")

# -----------------------------
# Anomaly Detection
# -----------------------------
st.subheader("🚨 Anomaly Detection")

anomaly_df = detect_anomalies_iqr(df, metric_col)

anomaly_count = int(anomaly_df["is_anomaly"].sum())
anomaly_pct = (anomaly_count / len(anomaly_df)) * 100 if len(anomaly_df) > 0 else 0

if anomaly_count > 0:
    st.warning(
        f"{anomaly_count} unusual records were detected in {metric_col} "
        f"({anomaly_pct:.2f}% of data)."
    )
    st.dataframe(
        anomaly_df[anomaly_df["is_anomaly"]].head(20),
        use_container_width=True
    )
else:
    st.success("No major anomalies detected.")

# -----------------------------
# Visual analysis
# -----------------------------
st.subheader("📈 Visual Analysis")

tab_labels = ["Trend", "Distribution", "Category", "Relationship", "Detailed Tables"]

if text_col is not None:
    tab_labels.append("NLP Insights")

tabs = st.tabs(tab_labels)

tab1 = tabs[0]
tab2 = tabs[1]
tab3 = tabs[2]
tab4 = tabs[3]
tab5 = tabs[4]

if text_col is not None:
    tab6 = tabs[5]

with tab1:
    trend_df = df[[time_col, metric_col]].copy().sort_values(time_col)

    line_chart = alt.Chart(trend_df).mark_line(point=True).encode(
        x=alt.X(f"{time_col}:T", title="Date"),
        y=alt.Y(f"{metric_col}:Q", title=metric_col),
        tooltip=[alt.Tooltip(f"{time_col}:T", title="Date"), alt.Tooltip(f"{metric_col}:Q", format=".2f")]
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
    valid_category = False
    if category_col != "None" and category_col in df.columns:
        if df[category_col].dtype == "object" or str(df[category_col].dtype).startswith("category"):
            valid_category = True

    if valid_category:
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
            if df[second_category_col].dtype == "object" or str(df[second_category_col].dtype).startswith("category"):
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
        st.warning("Selected category is not categorical. Please choose a valid category column from the sidebar.")

with tab4:
    other_numeric = [c for c in detect_metric_candidates(df) if c != metric_col and c in df.columns]

    if other_numeric:
        x_col = st.selectbox("Select numeric X-axis", other_numeric, key="relationship_x")

        rel_df = df.copy()

        rel_df[x_col] = pd.to_numeric(rel_df[x_col], errors="coerce")
        rel_df[metric_col] = pd.to_numeric(rel_df[metric_col], errors="coerce")

        rel_df = rel_df[[x_col, metric_col]].dropna()
        rel_df = rel_df[np.isfinite(rel_df[x_col]) & np.isfinite(rel_df[metric_col])]

        if not rel_df.empty and len(rel_df) >= 2:
            chart_df = rel_df.rename(columns={
                x_col: "x_value",
                metric_col: "y_value"
            })

            scatter = alt.Chart(chart_df).mark_circle(size=80, opacity=0.65).encode(
                x=alt.X("x_value:Q", title=x_col),
                y=alt.Y("y_value:Q", title=metric_col),
                tooltip=[
                    alt.Tooltip("x_value:Q", title=x_col, format=".2f"),
                    alt.Tooltip("y_value:Q", title=metric_col, format=".2f")
                ]
            ).properties(height=380)

            st.altair_chart(scatter, use_container_width=True)
        else:
            st.info("Not enough valid numeric data is available for relationship analysis.")
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

    valid_category = False
    if category_col != "None" and category_col in df.columns:
        if df[category_col].dtype == "object" or str(df[category_col].dtype).startswith("category"):
            valid_category = True

    if valid_category:
        st.markdown("#### Category Summary")
        cat_summary = (
            df.groupby(category_col)[metric_col]
            .agg(["sum", "mean", "count"])
            .sort_values("sum", ascending=False)
            .reset_index()
        )
        cat_summary.columns = [category_col, f"Total {metric_col}", f"Average {metric_col}", "Records"]
        st.dataframe(cat_summary, use_container_width=True)
if text_col is not None:
    with tab6:
        st.subheader("🧠 Advanced NLP Insights")

        n1, n2, n3 = st.columns(3)
        total_reviews = len(df_raw)
        avg_review_length = df_raw["review_length"].mean() if "review_length" in df_raw.columns else 0
        avg_rating = df_raw["rating"].mean() if "rating" in df_raw.columns else None

        n1.metric("Total Reviews", total_reviews)
        n2.metric("Average Review Length", f"{avg_review_length:.1f} words")
        n3.metric("Average Rating", f"{avg_rating:.2f}" if avg_rating is not None else "N/A")

        st.markdown("#### Sentiment Summary")
        if "sentiment_label" in df_raw.columns:
            sentiment_counts = df_raw["sentiment_label"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]

            sentiment_chart = alt.Chart(sentiment_counts).mark_bar().encode(
                x=alt.X("Sentiment:N", title="Sentiment"),
                y=alt.Y("Count:Q", title="Number of Reviews"),
                tooltip=["Sentiment", "Count"]
            ).properties(height=300)

            st.altair_chart(sentiment_chart, use_container_width=True)

        st.markdown("#### TF-IDF Top Keywords")
        keyword_df, _, _ = build_tfidf_keywords(df_raw["clean_text"], top_n=15)
        if not keyword_df.empty:
            keyword_chart = alt.Chart(keyword_df).mark_bar().encode(
                x=alt.X("Score:Q", title="Average TF-IDF Score"),
                y=alt.Y("Keyword:N", sort="-x", title="Keyword"),
                tooltip=["Keyword", alt.Tooltip("Score:Q", format=".4f")]
            ).properties(height=420)

            st.altair_chart(keyword_chart, use_container_width=True)
        else:
            st.info("Not enough text data to extract keywords.")

        st.markdown("#### Review Length Distribution")
        if "review_length" in df_raw.columns:
            length_chart = alt.Chart(df_raw).mark_bar().encode(
                x=alt.X("review_length:Q", bin=alt.Bin(maxbins=30), title="Review Length (words)"),
                y=alt.Y("count()", title="Count")
            ).properties(height=300)

            st.altair_chart(length_chart, use_container_width=True)

        if "rating" in df_raw.columns and "review_length" in df_raw.columns:
            st.markdown("#### Rating vs Review Length")
            scatter_df = df_raw[["rating", "review_length"]].copy()
            scatter_df["rating"] = pd.to_numeric(scatter_df["rating"], errors="coerce")
            scatter_df["review_length"] = pd.to_numeric(scatter_df["review_length"], errors="coerce")
            scatter_df = scatter_df.dropna()

            if not scatter_df.empty:
                rating_len_chart = alt.Chart(scatter_df).mark_circle(size=60, opacity=0.5).encode(
                    x=alt.X("review_length:Q", title="Review Length"),
                    y=alt.Y("rating:Q", title="Rating"),
                    tooltip=["review_length", alt.Tooltip("rating:Q", format=".2f")]
                ).properties(height=320)

                st.altair_chart(rating_len_chart, use_container_width=True)

        st.markdown("#### Topic Clusters")
        cluster_count = st.slider("Number of topic clusters", min_value=2, max_value=6, value=3, key="cluster_slider")

        clustered_df, top_terms_per_cluster, _ = cluster_reviews(df_raw["clean_text"], n_clusters=cluster_count)

        if clustered_df is not None:
            cluster_summary = clustered_df["cluster"].value_counts().sort_index().reset_index()
            cluster_summary.columns = ["Cluster", "Count"]

            cluster_chart = alt.Chart(cluster_summary).mark_bar().encode(
                x=alt.X("Cluster:N", title="Cluster"),
                y=alt.Y("Count:Q", title="Reviews"),
                tooltip=["Cluster", "Count"]
            ).properties(height=300)

            st.altair_chart(cluster_chart, use_container_width=True)

            st.markdown("#### Cluster Themes")
            for cluster_id, terms in top_terms_per_cluster.items():
                st.write(f"**Cluster {cluster_id}:** {', '.join(terms)}")

            st.markdown("#### Sample Reviews by Cluster")
            sample_cluster = st.selectbox(
                "Select a cluster to inspect",
                sorted(clustered_df["cluster"].unique().tolist()),
                key="sample_cluster_select"
            )

            cluster_samples = clustered_df[clustered_df["cluster"] == sample_cluster].head(10).copy()

            if title_col is not None and title_col in df_raw.columns:
                original_texts = df_raw[[title_col, text_col]].copy().reset_index(drop=True)
                display_df = pd.concat(
                    [cluster_samples.reset_index(drop=True), original_texts.iloc[:len(clustered_df)].reset_index(drop=True)],
                    axis=1
                )
                display_df = display_df[display_df["cluster"] == sample_cluster].head(10)
                st.dataframe(display_df[[title_col, text_col, "cluster"]], use_container_width=True)
            else:
                st.dataframe(cluster_samples[["clean_text", "cluster"]], use_container_width=True)
        else:
            st.info("Not enough text data to build topic clusters.")

        st.markdown("#### Positive vs Negative Keywords")
        if "sentiment_label" in df_raw.columns:
            positive_df = df_raw[df_raw["sentiment_label"] == "Positive"]
            negative_df = df_raw[df_raw["sentiment_label"] == "Negative"]

            pos_words = get_word_counts(positive_df["clean_text"], stopwords=basic_stopwords, top_n=10)
            neg_words = get_word_counts(negative_df["clean_text"], stopwords=basic_stopwords, top_n=10)

            col_pos, col_neg = st.columns(2)

            with col_pos:
                st.markdown("**Top Positive Words**")
                if pos_words:
                    st.dataframe(pd.DataFrame(pos_words, columns=["Word", "Count"]), use_container_width=True)
                else:
                    st.info("No strong positive keywords found.")

            with col_neg:
                st.markdown("**Top Negative Words**")
                if neg_words:
                    st.dataframe(pd.DataFrame(neg_words, columns=["Word", "Count"]), use_container_width=True)
                else:
                    st.info("No strong negative keywords found.")

        st.markdown("#### Search Reviews")
        query = st.text_input("Search for a keyword in reviews", key="review_search")
        if query:
            matched = df_raw[df_raw["clean_text"].str.contains(query.lower(), na=False)]
            if title_col is not None:
                st.dataframe(matched[[title_col, text_col]].head(20), use_container_width=True)
            else:
                st.dataframe(matched[[text_col]].head(20), use_container_width=True)
        
    
        

st.divider()

# -----------------------------
# Downloads
# -----------------------------
st.subheader("📥 Downloads")

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

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "This dashboard is built for flexible business file analysis, interactive filtering, and simple machine-learning-based forecasting."
)

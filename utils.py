import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression


def inject_css():
    st.markdown("""
    <style>
    :root {
        --bg: #07111f;
        --panel: #0f172a;
        --panel-2: #111827;
        --text: #f8fafc;
        --muted: #94a3b8;
        --accent: #3b82f6;
        --border: rgba(148, 163, 184, 0.14);
        --shadow: 0 16px 40px rgba(0, 0, 0, 0.28);
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(59,130,246,0.12), transparent 28%),
            linear-gradient(180deg, #07111f 0%, #0b1220 100%);
        color: var(--text);
    }
    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .section-card {
        background: linear-gradient(180deg, rgba(17,24,39,0.96), rgba(15,23,42,0.98));
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 18px;
        box-shadow: var(--shadow);
        margin-bottom: 14px;
    }
    .info-card {
        background: linear-gradient(180deg, rgba(30,41,59,0.95), rgba(15,23,42,0.98));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.20);
        margin-bottom: 12px;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(180deg, rgba(30,41,59,0.96), rgba(15,23,42,0.99));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 18px 16px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
        border-right: 1px solid var(--border);
    }
    </style>
    """, unsafe_allow_html=True)


def section_title(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"<p style='color:#94a3b8'>{subtitle}</p>", unsafe_allow_html=True)


def card_open():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)


def card_close():
    st.markdown('</div>', unsafe_allow_html=True)


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
    outlet_sizes = np.random.choice(["Small", "Medium", "High"], size=n)
    regions = np.random.choice(["North", "South", "East", "West"], size=n)

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
    priorities = ["sales", "revenue", "profit", "amount", "income", "total", "value", "quantity", "cost"]
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

        if df[col].dtype == "object" or unique_count < 50:
            col_lower = col.lower()
            if any(k in col_lower for k in ["category", "type", "region", "segment", "product", "item", "outlet", "store"]):
                preferred.append(col)
            else:
                fallback.append(col)

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


def moving_average_forecast(series, window=5):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < window:
        return None
    return float(series.tail(window).mean())


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


def apply_plotly_theme(fig, height=420):
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.95)",
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color="#f8fafc")
    )
    return fig


def prepare_state(df_raw):
    df_raw = normalize_columns(df_raw)
    df_raw["generated_date"] = pd.date_range(start="2024-01-01", periods=len(df_raw), freq="D")

    text_col = None
    title_col = None

    if "review_content" in df_raw.columns:
        text_col = "review_content"
    elif "about_product" in df_raw.columns:
        text_col = "about_product"

    if "review_title" in df_raw.columns:
        title_col = "review_title"

    if text_col is not None:
        df_raw["clean_text"] = df_raw[text_col].fillna("").astype(str).apply(clean_text)
        df_raw["review_length"] = df_raw["clean_text"].apply(lambda x: len(x.split()))

    if "rating" in df_raw.columns:
        df_raw["rating"] = pd.to_numeric(df_raw["rating"], errors="coerce")
        df_raw["sentiment_label"] = df_raw["rating"].apply(simple_sentiment_from_rating)

    numeric_candidates = detect_metric_candidates(df_raw)
    detected_date_col = detect_date_column(df_raw)
    suggested_metric = suggest_main_metric(numeric_candidates)
    suggested_profit = suggest_profit_column(df_raw.columns.tolist())
    suggested_cost = suggest_cost_column(df_raw.columns.tolist())
    suggested_quantity = suggest_quantity_column(df_raw.columns.tolist())

    category_candidates = suggest_category_columns(
        df_raw,
        exclude_cols=[c for c in [suggested_metric, detected_date_col, suggested_profit, suggested_cost, suggested_quantity] if c]
    )

    return {
        "df_raw": df_raw,
        "text_col": text_col,
        "title_col": title_col,
        "numeric_candidates": numeric_candidates,
        "detected_date_col": detected_date_col,
        "suggested_metric": suggested_metric,
        "suggested_profit": suggested_profit,
        "suggested_cost": suggested_cost,
        "suggested_quantity": suggested_quantity,
        "category_candidates": category_candidates,
    }

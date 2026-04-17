import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="AI Business Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Business Intelligence Dashboard")
st.markdown("### Smart insights to grow your business")
st.write("Upload your business sales data and get insights, trends, category analysis, and AI-based forecasts.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Support Big Mart dataset by renaming Item_Outlet_Sales to sales
        if "Item_Outlet_Sales" in df.columns and "sales" not in df.columns:
            df.rename(columns={"Item_Outlet_Sales": "sales"}, inplace=True)

        # Add a generated date column if missing
        if "date" not in df.columns:
            df["date"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")

        st.subheader("Data Preview")
        st.dataframe(df.head())

        if "sales" not in df.columns:
            st.error("The CSV file must contain a 'sales' column or 'Item_Outlet_Sales' column.")
        else:
            # Clean sales
            df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["sales", "date"])
            df = df.sort_values("date").reset_index(drop=True)

            if df.empty:
                st.error("No valid data found after cleaning the file.")
            else:
                # Metrics
                total_sales = df["sales"].sum()
                avg_sales = df["sales"].mean()
                max_sales = df["sales"].max()

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Sales", f"€{total_sales:.2f}")
                col2.metric("Average Sales", f"€{avg_sales:.2f}")
                col3.metric("Best Sales Value", f"€{max_sales:.2f}")

                # Key insight
                st.subheader("📌 Key Insight")
                if df["sales"].iloc[-1] < avg_sales:
                    st.warning("Sales are below average. Immediate action may be needed.")
                else:
                    st.success("Sales are performing well compared to average.")

                # Sales trend
                st.subheader("Sales Trend")
                st.line_chart(df.set_index("date")["sales"])

                # Business insight
                st.subheader("Business Insight")
                if len(df) >= 2:
                    if df["sales"].iloc[-1] < df["sales"].iloc[-2]:
                        st.warning("Recent sales show a decline. You may need promotions or customer re-engagement.")
                    else:
                        st.success("Sales trend looks stable or improving.")
                else:
                    st.info("Add more sales records to detect a recent trend.")

                # AI forecast with Linear Regression
                st.subheader("🤖 AI Forecast (Linear Regression)")
                if len(df) >= 2:
                    X = np.arange(len(df)).reshape(-1, 1)
                    y = df["sales"].values

                    model = LinearRegression()
                    model.fit(X, y)

                    next_index = np.array([[len(df)]])
                    prediction = model.predict(next_index)[0]

                    st.info(f"Estimated next sales value: €{prediction:.2f}")

                    slope = model.coef_[0]
                    if slope > 0:
                        st.success("The model detects an upward sales trend.")
                    elif slope < 0:
                        st.warning("The model detects a downward sales trend.")
                    else:
                        st.info("The model detects a relatively flat sales trend.")
                else:
                    st.info("At least 2 rows are needed for AI forecasting.")

                # Product/category insights for Big Mart-style datasets
                if "Item_Type" in df.columns:
                    st.subheader("Top Product Categories")
                    category_sales = (
                        df.groupby("Item_Type")["sales"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(10)
                    )
                    st.bar_chart(category_sales)

                if "Outlet_Type" in df.columns:
                    st.subheader("Sales by Outlet Type")
                    outlet_sales = (
                        df.groupby("Outlet_Type")["sales"]
                        .sum()
                        .sort_values(ascending=False)
                    )
                    st.bar_chart(outlet_sales)

                if "Outlet_Size" in df.columns:
                    st.subheader("Sales by Outlet Size")
                    size_sales = (
                        df.groupby("Outlet_Size")["sales"]
                        .sum()
                        .sort_values(ascending=False)
                    )
                    st.bar_chart(size_sales)

                # Download cleaned report
                download_df = df.copy()
                download_df["date"] = download_df["date"].dt.strftime("%Y-%m-%d")

                csv = download_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Report",
                    data=csv,
                    file_name="report.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a CSV file to begin.")
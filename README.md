## 📸 Dashboard Preview

![Dashboard](C:\Users\Abdoulie Bah\Pictures\Screenshots/dashboard.png)

# 🤖 AI Business Intelligence Dashboard

An advanced **AI-powered business analytics dashboard** built with Python and Streamlit.  
This application automatically analyzes any business dataset, generates insights, detects anomalies, and provides forecasting using machine learning and NLP.

---

## 🚀 Features

### 📊 Data Analysis
- Upload CSV or Excel business datasets
- Automatic detection of:
  - Revenue / Sales / Profit columns
  - Date columns (auto-generated if missing)
  - Category variables
- Interactive filtering (date, category, value range)

### 📈 Forecasting
- Linear Regression forecasting
- Moving Average forecasting
- Trend detection (growth / decline)

### 💡 AI Insights
- Automatic business insights generation
- Performance evaluation (growth, stability, volatility)
- Category performance comparison
- Profitability analysis

### 🚨 Anomaly Detection
- Detect unusual records using IQR method
- Highlight outliers and abnormal patterns
- Percentage anomaly reporting

### 🧠 NLP (Text Analytics)
- Works with review-based datasets
- Text cleaning and preprocessing
- Sentiment analysis (Positive / Neutral / Negative)
- TF-IDF keyword extraction
- Topic clustering (KMeans)
- Review search functionality
- Positive vs Negative keyword analysis

### 📊 Visualization
- Trend charts
- Distribution analysis
- Category breakdown
- Relationship analysis
- NLP visual insights

### 📥 Export
- Download cleaned dataset
- Download summary report

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas & NumPy
- Scikit-learn (ML models)
- Altair (visualization)
- NLP (TF-IDF, clustering)

---

## ⚙️ How It Works

1. Upload a CSV or Excel dataset
2. The system automatically detects key business columns
3. Data is cleaned and processed
4. Insights, KPIs, and forecasts are generated
5. Optional NLP analysis is applied (if text data exists)
6. Results are visualized and available for download

---

## 📂 Supported Data

The dashboard works with:

- Sales / Revenue datasets
- Financial datasets
- E-commerce data
- Customer review datasets
- Big Mart dataset
- Any structured CSV/Excel file

If a date column is missing, the system automatically generates one.

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ai-business-intelligence-dashboard.git
cd ai-business-intelligence-dashboard

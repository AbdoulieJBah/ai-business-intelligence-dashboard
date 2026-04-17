# AI Business Intelligence Dashboard

An AI-powered business intelligence dashboard built with Python and Streamlit. This application allows users to upload business sales data, view key insights, analyze trends, and generate future sales forecasts using Machine Learning.

## Features

- Upload CSV business data
- Automatic sales column support
- Works with standard `sales` datasets and Big Mart datasets
- Sales trend visualization
- Key business insights
- AI-based sales forecasting using Linear Regression
- Product category analysis
- Outlet type and outlet size analysis
- Download cleaned report

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn

## How It Works

The dashboard:
1. Loads a CSV file
2. Cleans and processes the data
3. Detects the sales column
4. Generates summary metrics
5. Visualizes trends
6. Predicts the next sales value using Linear Regression

## Supported Datasets

The app supports:
- CSV files with a `sales` column
- Big Mart dataset with `Item_Outlet_Sales`

If a `date` column is missing, the app automatically generates one.

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ai-business-intelligence-dashboard.git
cd ai-business-intelligence-dashboard
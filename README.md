# 📺 YouTube Ad Revenue Prediction

## 📌 Project Overview

A machine learning project that predicts YouTube advertising revenue
(`ad_revenue_usd`) using video performance, engagement, audience,
device, country, and category-related features.

The project includes data cleaning, EDA, feature engineering,
regression modeling, model evaluation, and Streamlit deployment.

## 🛠️ Technologies

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Joblib
- Streamlit

## 🤖 Models Used

Five regression models were trained and compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regression
- Gradient Boosting Regression

### Best Model

**Linear Regression**

R²: **1.0000**

The model was selected based on its overall test performance.

## 🔍 Key Insights

- `watch_time_minutes` was the strongest predictive feature.
- Watch time and ad revenue showed a correlation of approximately **0.989**.
- Engagement-related features also contributed to revenue prediction.
- The dataset is synthetic, so the results represent patterns within
  this dataset rather than real-world causal relationships.

## 🚀 Streamlit App

The Streamlit application allows users to enter video information and
receive an estimated advertising revenue in **USD ($)**.

### Run the Application

```bash
pip install -r requirements.txt
streamlit run app.py
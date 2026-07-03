# 🛒 Shopper Spectrum
### Customer Intelligence & Product Recommendation Platform

An end-to-end E-Commerce Analytics platform that transforms raw retail transaction data into actionable business insights through customer segmentation, predictive analytics, and recommendation systems.

---

## 📌 Project Overview

Shopper Spectrum helps businesses understand customer purchasing behavior and make data-driven decisions by combining:

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- RFM Customer Segmentation
- KMeans Clustering
- Customer Health Scoring
- Product Recommendation Engine
- Interactive Streamlit Dashboard
- Business Intelligence Insights

---

## 🎯 Problem Statement

E-commerce businesses generate massive amounts of transactional data but often struggle to:

- Identify high-value customers
- Detect customers at risk of churn
- Understand purchasing patterns
- Recommend relevant products
- Make informed marketing decisions

Shopper Spectrum addresses these challenges through analytics and machine learning.

---

# 🏗️ Project Architecture

Raw Data
↓
Data Cleaning
↓
Feature Engineering
↓
RFM Analysis
↓
Customer Segmentation
↓
Product Recommendation System
↓
Business Insights
↓
Interactive Dashboard

---

# 📂 Project Structure

```text
SHOPPING SPECTRUM
│
├── Data
│   ├── online_retail.csv
│   ├── cleaned_retail.csv
│   ├── rfm_data.csv
│   └── customer_segments.csv
│
├── Models
│   ├── scaler.pkl
│   ├── kmeans.pkl
│   ├── similarity.pkl
│   └── products.pkl
│
├── Notebook
│   └── analysis.ipynb
│
├── app.py
├── data_cleaning.py
├── rfm_analysis.py
├── clustering.py
├── recommendation.py
├── requirements.txt
└── README.md
```

---

# 🛠️ Tech Stack

### Programming Language
- Python

### Data Analysis
- Pandas
- NumPy

### Data Visualization
- Plotly
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
- KMeans Clustering
- StandardScaler
- Cosine Similarity

### Dashboard
- Streamlit
- Streamlit Option Menu

---

# 📊 Dataset Information

Dataset: Online Retail Dataset

Records: 541,909 transactions

Features:

- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

---

# 🧹 Data Cleaning

Performed:

✔ Removed missing Customer IDs

✔ Removed missing product descriptions

✔ Removed duplicate records

✔ Removed cancelled transactions

✔ Removed invalid quantities and prices

✔ Created TotalPrice feature

---

# 📈 Exploratory Data Analysis

Visualizations performed:

- Transaction Volume by Country
- Top Selling Products
- Monthly Revenue Trend
- Revenue Distribution
- Top Customers by Revenue
- Segment Distribution
- Revenue by Segment
- Recency Distribution
- Frequency Distribution
- Monetary Distribution
- Correlation Heatmap
- Frequency vs Monetary Scatter Plot
- 3D Customer Visualization
- Elbow Method

---

# 👥 RFM Analysis

### Recency
How recently a customer made a purchase.

### Frequency
How often a customer purchases.

### Monetary
How much a customer spends.

Generated dataset:

```text
CustomerID
Recency
Frequency
Monetary
```

---

# 🤖 Customer Segmentation

Machine Learning Algorithm:

- StandardScaler
- KMeans Clustering

Silhouette Score:

```text
0.616
```

### Customer Segments

🏆 VIP Customers

💜 High-Value Customers

🟢 Regular Customers

🔴 At-Risk Customers

---

# ❤️ Customer Health Score

Each customer receives:

- R Score
- F Score
- M Score
- Combined RFM Score
- Health Score (0-100)

This enables customer prioritization and targeted marketing campaigns.

---

# 🎁 Product Recommendation System

Recommendation Engine:

- Customer-Product Matrix
- Cosine Similarity

Features:

✔ Product-to-product recommendations

✔ Top 5 similar products

✔ Real-time recommendations within dashboard

---

# 📊 Streamlit Dashboard Pages

### 🏠 Dashboard
- KPI Cards
- Monthly Revenue Trend
- Segment Distribution

### 📊 EDA
- Country Analysis
- Product Analysis
- Revenue Analysis

### 👥 Segmentation
- Cluster Distribution
- Scatter Visualizations
- Segment Metrics

### 🎯 Predictor
- Predict Customer Segment
- Customer Health Analysis

### 🎁 Recommendation
- Product Recommendation Engine

### 💡 Insights
- Strategic Business Recommendations

---

# 💡 Key Business Insights

### VIP Customers
Generate the majority of revenue despite representing a small customer base.

### High-Value Customers
Demonstrate strong purchasing behavior and are ideal candidates for loyalty programs.

### Regular Customers
Represent the largest customer base and offer significant upselling opportunities.

### At-Risk Customers
Require personalized campaigns and retention strategies.



# Future Improvements

- Real-time recommendation API
- Advanced customer lifetime value prediction
- Churn prediction model
- Cloud deployment
- Personalized marketing engine

---

# 👨‍💻 Author

**Shubham Samarpit**

Aspiring Data Analyst | Python | SQL | Power BI | Machine Learning | Data Visualization


import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from streamlit_option_menu import option_menu

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🧾",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(BASE_DIR, "Models")

CLUSTER_TO_SEGMENT = {0: "Regular", 1: "At-Risk", 2: "VIP", 3: "High-Value"}

# ---- design tokens -----------------------------------------------------
INK = "#eef1f5"
MUTED = "#8593a8"
BG = "#0b0f14"
SURFACE = "#121926"
LINE = "#2a3444"
GREEN = "#2fdd9a"    # register tape green
AMBER = "#ffb020"    # price-tag amber
BLUE = "#4fb3ff"     # ledger blue
RED = "#fb6a6a"      # markdown-down red

SEGMENT_INFO = {
    "VIP": {
        "color": GREEN,
        "characteristics": "Your best customers: very frequent, very recent, very high spend.",
        "action": "White-glove treatment. Dedicated support, early access, exclusive perks. Protect this group at all costs.",
    },
    "High-Value": {
        "color": BLUE,
        "characteristics": "Highly active, frequent buyers who spend premium amounts and bought recently.",
        "action": "Reward with VIP programs, exclusive access, and early product launches. Retain at all costs.",
    },
    "Regular": {
        "color": AMBER,
        "characteristics": "Steady purchasers with moderate order frequency and total spend.",
        "action": "Nurture with upsell recommendations, cross-sell opportunities, and loyalty programs.",
    },
    "At-Risk": {
        "color": RED,
        "characteristics": "Customers who haven't ordered in a very long time, with minimal spend.",
        "action": "Deploy win-back marketing emails, feedback questionnaires, or high-discount recovery deals.",
    },
}

CHART_COLORWAY = [GREEN, AMBER, BLUE, RED, "#c084fc", "#7dd3fc"]


# ==========================================================
# GLOBAL STYLE — receipt / ledger identity
# ==========================================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background-color: {BG};
        color: {INK};
    }}

    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.01em;
    }}

    /* hide default streamlit chrome */
    #MainMenu, footer, header {{visibility: hidden;}}

    /* ---- ticket header ---- */
    .ticket {{
        background: {SURFACE};
        border: 1px dashed {LINE};
        border-radius: 6px;
        padding: 22px 30px 16px 30px;
        margin-bottom: 6px;
    }}
    .ticket-top {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        flex-wrap: wrap;
        gap: 8px;
    }}
    .ticket-store {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 30px;
        font-weight: 700;
        letter-spacing: 0.02em;
        color: {INK};
    }}
    .ticket-meta {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        color: {MUTED};
        text-align: right;
    }}
    .ticket-tagline {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        color: {MUTED};
        margin-top: 4px;
    }}
    .barcode {{
        margin-top: 16px;
        height: 26px;
        background-image: repeating-linear-gradient(90deg,
            {LINE} 0px, {LINE} 2px, transparent 2px, transparent 4px,
            {LINE} 4px, {LINE} 5px, transparent 5px, transparent 9px,
            {LINE} 9px, {LINE} 12px, transparent 12px, transparent 15px,
            {LINE} 15px, {LINE} 16px, transparent 16px, transparent 21px);
        opacity: 0.55;
    }}

    /* ---- barcode section divider ---- */
    .divider {{
        height: 14px;
        margin: 26px 0 22px 0;
        background-image: repeating-linear-gradient(90deg,
            {LINE} 0px, {LINE} 1px, transparent 1px, transparent 3px,
            {LINE} 3px, {LINE} 4px, transparent 4px, transparent 8px,
            {LINE} 8px, {LINE} 9px, transparent 9px, transparent 12px);
        opacity: 0.35;
    }}

    /* ---- KPI receipt line items ---- */
    .kpi-row {{ display: flex; gap: 14px; flex-wrap: wrap; margin-top: 4px; }}
    .kpi-item {{
        flex: 1; min-width: 180px;
        background: {SURFACE};
        border: 1px solid {LINE};
        border-radius: 6px;
        padding: 14px 18px;
    }}
    .kpi-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {MUTED};
    }}
    .kpi-value {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 28px;
        font-weight: 700;
        margin-top: 6px;
        color: {INK};
    }}

    /* ---- price tag segment cards ---- */
    .price-tag {{
        position: relative;
        background: {SURFACE};
        border: 1px solid {LINE};
        border-radius: 4px 14px 14px 4px;
        padding: 14px 18px 14px 26px;
        margin-bottom: 14px;
    }}
    .price-tag::before {{
        content: '';
        position: absolute;
        left: 9px; top: 18px;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: {BG};
        border: 1px solid {LINE};
    }}
    .price-tag-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 6px;
    }}
    .price-tag-body {{ font-size: 13.5px; color: {INK}; line-height: 1.55; }}
    .price-tag-body b {{ color: {MUTED}; font-weight: 600; }}

    /* ---- recommendation stub cards ---- */
    .stub {{
        background: {SURFACE};
        border: 1px solid {LINE};
        border-top: 3px dashed {LINE};
        border-radius: 4px;
        padding: 14px 14px 12px 14px;
        min-height: 130px;
    }}
    .stub-name {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 13.5px;
        line-height: 1.3;
    }}
    .stub-score {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        color: {GREEN};
        margin-top: 10px;
    }}

    /* ---- option menu overrides ---- */
    .nav-link {{ font-family: 'Space Grotesk', sans-serif !important; }}

    /* ---- tabs as ticket stubs ---- */
    .stTabs [data-baseweb="tab"] {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
    }}

    /* dataframes */
    [data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 6px; }}

    /* footer receipt line */
    .foot {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11.5px;
        color: {MUTED};
        text-align: center;
        letter-spacing: 0.06em;
        margin-top: 6px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def divider():
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


def kpi_row(items):
    """items: list of (label, value) tuples"""
    html = "<div class='kpi-row'>"
    for label, value in items:
        html += (
            f"<div class='kpi-item'><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div></div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def style_fig(fig, title=None):
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="IBM Plex Mono, monospace", color=INK, size=12),
        colorway=CHART_COLORWAY,
        margin=dict(t=40 if title else 20, l=10, r=10, b=10),
        title=title,
    )
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE)
    return fig


def footer():
    divider()
    st.markdown(
        "<div class='foot'>*** SHOPPER SPECTRUM — STREAMLIT · SCIKIT-LEARN · PANDAS ***</div>",
        unsafe_allow_html=True,
    )


# ==========================================================
# DATA / MODEL LOADING
# ==========================================================
@st.cache_data
def load_csv(filename):
    return pd.read_csv(os.path.join(DATA_DIR, filename))


@st.cache_resource
def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


try:
    retail = load_csv("cleaned_retail.csv.gz")
    retail["InvoiceDate"] = pd.to_datetime(retail["InvoiceDate"])
    segments = load_csv("customer_segments.csv")
    rfm = load_csv("rfm_data.csv")

    scaler = load_model("scaler.pkl")
    kmeans = load_model("kmeans.pkl")
    similarity_df = load_model("similarity.pkl")
    products = load_model("products.pkl")

    LOAD_ERROR = None
except Exception as e:  # noqa: BLE001
    LOAD_ERROR = str(e)


# ==========================================================
# TICKET HEADER
# ==========================================================
st.markdown(
    f"""
    <div class='ticket'>
        <div class='ticket-top'>
            <div class='ticket-store'>🧾 SHOPPER SPECTRUM</div>
            <div class='ticket-meta'>
                RECEIPT NO. 0001-SS<br>
                {datetime.now().strftime('%d %b %Y · %H:%M')}
            </div>
        </div>
        <div class='ticket-tagline'>customer segmentation &amp; product recommendation — e-commerce</div>
        <div class='barcode'></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if LOAD_ERROR:
    st.error(
        "Could not load data/model files. Make sure `Data/` and `Models/` "
        f"folders sit next to app.py with the required files.\n\nDetails: {LOAD_ERROR}"
    )
    st.stop()

selected = option_menu(
    menu_title=None,
    options=[
        "Project Overview",
        "E-Commerce Insights",
        "Customer Segmentation",
        "Product Recommendation",
    ],
    icons=["file-earmark-text", "bar-chart", "people", "bullseye"],
    orientation="horizontal",
    styles={
        "container": {"padding": "6px 0", "background-color": SURFACE, "border": f"1px solid {LINE}",
                       "border-radius": "6px", "margin-top": "10px"},
        "icon": {"color": GREEN, "font-size": "14px"},
        "nav-link": {"font-size": "14px", "color": MUTED, "text-align": "center", "margin": "0px",
                      "padding": "10px 14px", "--hover-color": "#1a2230"},
        "nav-link-selected": {"background-color": GREEN, "color": BG, "font-weight": "700"},
    },
)

divider()


# ==========================================================
# TAB 1 — PROJECT OVERVIEW
# ==========================================================
if selected == "Project Overview":

    st.header("📋 Project Overview & Business Case")

    st.subheader("📢 Problem Statement")
    st.write(
        "The global e-commerce industry generates vast amounts of transaction data daily, "
        "offering valuable insights into customer purchasing behaviors. Analyzing this data "
        "is essential for identifying meaningful customer segments and recommending relevant "
        "products to enhance customer experience and drive business growth."
    )
    st.write(
        "This project examines transaction data from an online retail business to uncover "
        "patterns in customer purchase behavior, segment customers using Recency, Frequency, "
        "and Monetary (RFM) analysis, and build a product recommendation system using "
        "collaborative filtering."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Business Objectives")
        st.markdown(
            """
            - **Understand customer buying behavior** to support better business decisions.
            - **Find and reward valuable customers** (VIP / High-Value segments).
            - **Personalize product recommendations** to improve the shopping experience.
            - **Reduce customer churn** by identifying and re-engaging at-risk shoppers.
            """
        )

    with col2:
        st.subheader("📌 Real-time Business Use Cases")
        st.info(
            """
            1. **Targeted Marketing** — tailor campaigns to specific customer segments.
            2. **Personalized Upselling** — suggest products bought alongside past purchases.
            3. **Customer Retention** — identify at-risk clients for win-back campaigns.
            4. **Stock Optimization** — track product popularity for inventory planning.
            """
        )

    divider()

    # summary "receipt" of the dataset itself
    st.subheader("🧾 Dataset Receipt")
    kpi_row([
        ("Transactions", f"{len(retail):,}"),
        ("Unique Customers", f"{retail['CustomerID'].nunique():,}"),
        ("Unique Products", f"{retail['Description'].nunique():,}"),
        ("Countries", f"{retail['Country'].nunique():,}"),
    ])

    divider()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📊 Dataset Details")
        st.caption("Real transactions from a UK-based online retail store.")
        st.dataframe(
            pd.DataFrame(
                {
                    "Column": [
                        "InvoiceNo", "StockCode", "Description", "Quantity",
                        "InvoiceDate", "UnitPrice", "CustomerID", "Country",
                    ],
                    "Meaning": [
                        "Transaction bill number (prefixed 'C' if cancelled)",
                        "Unique product/item code",
                        "Product name / item description",
                        "Units purchased per transaction",
                        "Date and time of the transaction",
                        "Price of a single unit",
                        "Unique customer identifier",
                        "Country the customer is based in",
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    with col4:
        st.subheader("⚙️ Project Flow")
        st.markdown(
            """
            1. **Load & Explore** — read transaction records, examine structure.
            2. **Data Cleaning** — remove duplicates, missing CustomerIDs, cancelled orders,
               and invalid quantities/prices.
            3. **Exploratory Data Analysis** — sales trends, top countries, top products.
            4. **RFM Feature Engineering** — Recency, Frequency, Monetary per customer.
            5. **Unsupervised Clustering** — standardize RFM, run KMeans (k=4) for segments.
            6. **Collaborative Recommendation** — item-based cosine similarity, top-5 products.
            """
        )

    footer()


# ==========================================================
# TAB 2 — E-COMMERCE INSIGHTS
# ==========================================================
elif selected == "E-Commerce Insights":

    st.header("📊 E-Commerce Insights & Exploratory Data Analysis")
    st.caption("Explore business metrics, sales trends, and customer patterns interactively.")

    with st.expander("🔍 Filter Dashboard Data", expanded=True):
        fc1, fc2 = st.columns([1, 2])

        with fc1:
            countries = st.multiselect(
                "Select Countries (leave empty for All)",
                sorted(retail["Country"].unique()),
            )

        with fc2:
            min_date = retail["InvoiceDate"].min().date()
            max_date = retail["InvoiceDate"].max().date()
            date_range = st.slider(
                "Select Date Range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
            )

    filtered = retail.copy()
    if countries:
        filtered = filtered[filtered["Country"].isin(countries)]
    filtered = filtered[
        (filtered["InvoiceDate"].dt.date >= date_range[0])
        & (filtered["InvoiceDate"].dt.date <= date_range[1])
    ]

    total_revenue = filtered["TotalPrice"].sum()
    total_orders = filtered["InvoiceNo"].nunique()
    active_customers = filtered["CustomerID"].nunique()
    avg_order_value = total_revenue / total_orders if total_orders else 0

    kpi_row([
        ("Total Revenue", f"£{total_revenue:,.0f}"),
        ("Total Orders", f"{total_orders:,}"),
        ("Active Customers", f"{active_customers:,}"),
        ("Avg. Order Value", f"£{avg_order_value:,.2f}"),
    ])

    divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📈 Monthly Revenue Trend")
        monthly = filtered.copy()
        monthly["YearMonth"] = monthly["InvoiceDate"].dt.to_period("M").astype(str)
        monthly_sales = monthly.groupby("YearMonth")["TotalPrice"].sum().reset_index()
        fig = px.area(monthly_sales, x="YearMonth", y="TotalPrice")
        fig.update_traces(line_color=GREEN, fillcolor="rgba(47,221,154,0.18)")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with c2:
        st.subheader("🌍 Top 10 Countries by Revenue")
        top_countries = (
            filtered.groupby("Country")["TotalPrice"].sum()
            .sort_values(ascending=False).head(10).reset_index()
        )
        fig = px.bar(top_countries, x="Country", y="TotalPrice",
                     color_discrete_sequence=[BLUE])
        st.plotly_chart(style_fig(fig), use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("📦 Top 10 Best-Selling Products")
        top_products = (
            filtered.groupby("Description")["Quantity"].sum()
            .sort_values(ascending=False).head(10).reset_index()
        )
        fig = px.bar(top_products, x="Description", y="Quantity",
                     color_discrete_sequence=[AMBER])
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with c4:
        st.subheader("👥 Top 10 Customers by Spend")
        top_customers = (
            filtered.groupby("CustomerID")["TotalPrice"].sum()
            .sort_values(ascending=False).head(10).reset_index()
        )
        top_customers["CustomerID"] = "Customer " + top_customers["CustomerID"].astype(int).astype(str)
        fig = px.bar(top_customers, x="CustomerID", y="TotalPrice",
                     color_discrete_sequence=[GREEN])
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    divider()
    st.subheader("🧑‍🤝‍🧑 Customer Purchase Behavior")

    b1, b2 = st.columns([2, 1])

    orders_per_customer = filtered.groupby("CustomerID")["InvoiceNo"].nunique()

    with b1:
        st.caption("Distribution of Orders per Customer")
        dist = orders_per_customer.value_counts().sort_index().reset_index()
        dist.columns = ["Orders", "NumCustomers"]
        dist["Orders"] = dist["Orders"].apply(lambda x: str(x) if x < 20 else "20+")
        dist = dist.groupby("Orders", sort=False)["NumCustomers"].sum().reset_index()
        fig = px.bar(dist, x="Orders", y="NumCustomers", color_discrete_sequence=[BLUE])
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with b2:
        st.caption("Order Statistics")
        stats = pd.DataFrame(
            {
                "Metric": ["Average Orders/Customer", "Median Orders/Customer",
                           "Min Orders", "Max Orders", "Standard Deviation"],
                "Value": [
                    f"{orders_per_customer.mean():.2f}",
                    f"{orders_per_customer.median():.0f}",
                    f"{orders_per_customer.min():.0f}",
                    f"{orders_per_customer.max():.0f}",
                    f"{orders_per_customer.std():.2f}",
                ],
            }
        )
        st.dataframe(stats, hide_index=True, use_container_width=True)

    footer()


# ==========================================================
# TAB 3 — CUSTOMER SEGMENTATION
# ==========================================================
elif selected == "Customer Segmentation":

    st.header("👥 Customer Segmentation & RFM Analysis")
    st.caption("Group customers based on Recency, Frequency, and Monetary value using KMeans Clustering.")

    sub_tab1, sub_tab2 = st.tabs(["🎯 Predict Customer Segment", "📊 RFM Clustering & Elbow Method"])

    # ---------------- Predict Customer Segment ----------------
    with sub_tab1:
        st.subheader("Predict a Customer's Segment")
        st.write("Enter a customer's RFM values to see which segment they fall into.")

        p1, p2, p3 = st.columns(3)
        with p1:
            recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
        with p2:
            frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5)
        with p3:
            monetary = st.number_input("Monetary (total amount spent)", min_value=0.0, value=500.0, step=10.0)

        if st.button("Predict Segment", type="primary"):
            sample = pd.DataFrame({"Recency": [recency], "Frequency": [frequency], "Monetary": [monetary]})
            scaled = scaler.transform(sample)
            cluster = kmeans.predict(scaled)[0]
            segment = CLUSTER_TO_SEGMENT.get(cluster, "Unknown")
            info = SEGMENT_INFO.get(segment, {})
            st.markdown(
                f"""
                <div class='price-tag' style='border-left:4px solid {info.get("color", GREEN)};'>
                    <div class='price-tag-title'>Predicted Segment: {segment}</div>
                    <div class='price-tag-body'><b>Characteristics:</b> {info.get('characteristics','')}</div>
                    <div class='price-tag-body'><b>Action:</b> {info.get('action','')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------------- RFM Clustering & Elbow Method ----------------
    with sub_tab2:
        st.subheader("Clustering Methodology & RFM Analysis")
        st.write(
            "Customers are segmented using the **K-Means Clustering** algorithm applied to "
            "scaled Recency, Frequency, and Monetary (RFM) metrics."
        )

        @st.cache_data
        def compute_elbow_silhouette(_rfm):
            X = scaler.transform(_rfm[["Recency", "Frequency", "Monetary"]])
            ks = list(range(2, 9))
            inertias, sils = [], []
            sample = X if len(X) <= 5000 else X[np.random.RandomState(42).choice(len(X), 5000, replace=False)]
            for k in ks:
                km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
                inertias.append(km.inertia_)
                sils.append(silhouette_score(sample, KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(sample)))
            return ks, inertias, sils

        ks, inertias, sils = compute_elbow_silhouette(rfm)

        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**K-Means Elbow Method**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers", name="Inertia",
                                      line=dict(color=GREEN)))
            fig.update_layout(xaxis_title="Number of Clusters (k)", yaxis_title="Inertia")
            st.plotly_chart(style_fig(fig), use_container_width=True)

            st.markdown("**Silhouette Score**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ks, y=sils, mode="lines+markers", name="Silhouette",
                                      line=dict(color=AMBER)))
            fig.update_layout(xaxis_title="Number of Clusters (k)", yaxis_title="Score")
            st.plotly_chart(style_fig(fig), use_container_width=True)

        with e2:
            st.markdown("**RFM Customer Clusters**")
            merged = rfm.merge(segments[["CustomerID", "Cluster"]], on="CustomerID", how="left")
            fig = px.scatter(
                merged, x="Recency", y="Monetary", color=merged["Cluster"].astype(str),
                labels={"color": "Cluster"},
            )
            st.plotly_chart(style_fig(fig, title="Customer Segments (Recency vs Monetary)"),
                             use_container_width=True)

        st.markdown("**Silhouette Scores by Cluster Count (k):**")
        score_table = pd.DataFrame({"Number of Clusters (k)": ks, "Silhouette Score": [round(s, 4) for s in sils]})
        st.dataframe(score_table, hide_index=True, use_container_width=True)

        divider()
        st.subheader("📁 Segment Profiles & Statistics")

        profile = (
            segments.groupby("Segment")
            .agg(
                Customer_Count=("CustomerID", "count"),
                Avg_Recency=("Recency", "mean"),
                Avg_Frequency=("Frequency", "mean"),
                Avg_Monetary=("Monetary", "mean"),
            )
            .round(1)
            .reset_index()
            .sort_values("Avg_Monetary", ascending=False)
        )
        profile.columns = ["Segment", "Customer Count", "Average Recency (Days)",
                            "Average Frequency (Orders)", "Average Monetary Spend (£)"]
        st.dataframe(profile, hide_index=True, use_container_width=True)

        divider()
        st.subheader("🔎 Segment Profiles & Business Action Plans")

        seg_cols = st.columns(2)
        for i, (seg_name, info) in enumerate(SEGMENT_INFO.items()):
            if seg_name not in segments["Segment"].unique():
                continue
            with seg_cols[i % 2]:
                st.markdown(
                    f"""
                    <div class='price-tag' style='border-left:4px solid {info['color']};'>
                        <div class='price-tag-title'>{seg_name}</div>
                        <div class='price-tag-body'><b>Characteristics:</b> {info['characteristics']}</div>
                        <div class='price-tag-body'><b>Action:</b> {info['action']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    footer()


# ==========================================================
# TAB 4 — PRODUCT RECOMMENDATION
# ==========================================================
elif selected == "Product Recommendation":

    st.header("🎯 Find Similar Products")
    st.write("Enter a product name to get the top 5 recommended products, based on what customers who bought it also tend to buy.")

    product = st.selectbox("Choose or search a product", sorted(products))

    if st.button("Get Recommendations", type="primary"):
        recommendations = similarity_df[product].sort_values(ascending=False).iloc[1:6]

        st.markdown(
            f"""
            <div class='price-tag' style='border-left:4px solid {GREEN};'>
                <div class='price-tag-body'>Because you looked at: <b>{product}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cols = st.columns(5)
        for col, (item, score) in zip(cols, recommendations.items()):
            with col:
                st.markdown(
                    f"""
                    <div class='stub'>
                        <div class='stub-name'>{item}</div>
                        <div class='stub-score'>SIMILARITY&nbsp;{score:.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    footer()

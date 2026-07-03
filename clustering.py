"""
Shopper Spectrum - Customer Segmentation
----------------------------------------
Creates customer segments using KMeans clustering.

Input:
    Data/rfm_data.csv

Outputs:
    Data/customer_segments.csv
    Models/scaler.pkl
    Models/kmeans.pkl
"""

import os
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ======================================
# FILE PATHS
# ======================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RFM_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "rfm_data.csv"
)

SEGMENT_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "customer_segments.csv"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "Models",
    "scaler.pkl"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Models",
    "kmeans.pkl"
)

# ======================================
# LOAD DATA
# ======================================

def load_data():
    """Load RFM dataset."""
    return pd.read_csv(RFM_PATH)


# ======================================
# CUSTOMER SEGMENTATION
# ======================================

def create_segments(df):

    rfm = df.copy()

    features = [
        "Recency",
        "Frequency",
        "Monetary"
    ]

    X = rfm[features]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train KMeans
    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)

    rfm["Cluster"] = clusters

    # Silhouette Score
    score = silhouette_score(
        X_scaled,
        clusters
    )

    print(f"\nSilhouette Score: {score:.3f}")

    return rfm, scaler, kmeans


# ======================================
# CLUSTER PROFILE
# ======================================

def show_cluster_profile(df):

    profile = (
        df.groupby("Cluster")
        [
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
        .mean()
        .round(2)
    )

    print("\nCluster Profile")
    print(profile)

    return profile


# ======================================
# MAIN FUNCTION
# ======================================

def main():

    print("=" * 55)
    print("SHOPPER SPECTRUM - CUSTOMER SEGMENTATION")
    print("=" * 55)

    # Load data
    df = load_data()

    print(f"\nRFM Shape: {df.shape}")

    # Create clusters
    segments, scaler, model = create_segments(df)

    # Show cluster statistics
    show_cluster_profile(segments)

    # ==================================
    # Segment Names
    # ==================================

    segment_map = {
        0: "Regular",
        1: "At-Risk",
        2: "VIP",
        3: "High-Value"
    }

    segments["Segment"] = (
        segments["Cluster"]
        .map(segment_map)
    )

    # ==================================
    # R Score
    # ==================================

    segments["R_Score"] = pd.qcut(
        segments["Recency"],
        5,
        labels=[5, 4, 3, 2, 1]
    )

    # ==================================
    # F Score
    # ==================================

    segments["F_Score"] = pd.qcut(
        segments["Frequency"].rank(
            method="first"
        ),
        5,
        labels=[1, 2, 3, 4, 5]
    )

    # ==================================
    # M Score
    # ==================================

    segments["M_Score"] = pd.qcut(
        segments["Monetary"],
        5,
        labels=[1, 2, 3, 4, 5]
    )

    # ==================================
    # Combined RFM Score
    # ==================================

    segments["RFM_Score"] = (
        segments["R_Score"].astype(str)
        + segments["F_Score"].astype(str)
        + segments["M_Score"].astype(str)
    )

    # ==================================
    # Customer Health Score
    # ==================================

    segments["Health_Score"] = (
        (
            segments["R_Score"].astype(int)
            + segments["F_Score"].astype(int)
            + segments["M_Score"].astype(int)
        )
        / 15
        * 100
    ).round(0).astype(int)

    # ==================================
    # Save Dataset
    # ==================================

    segments.to_csv(
        SEGMENT_PATH,
        index=False
    )

    # Save Models
    joblib.dump(
        scaler,
        SCALER_PATH
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    # ==================================
    # Summary
    # ==================================

    print("\nSegment Distribution")
    print(
        segments["Segment"]
        .value_counts()
    )

    print(f"\nSaved Dataset : {SEGMENT_PATH}")
    print(f"Saved Scaler : {SCALER_PATH}")
    print(f"Saved Model : {MODEL_PATH}")


# ======================================
# RUN PROGRAM
# ======================================

if __name__ == "__main__":
    main()
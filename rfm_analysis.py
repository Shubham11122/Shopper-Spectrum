"""
Shopper Spectrum - Step 2: RFM Analysis
--------------------------------------
Creates RFM (Recency, Frequency, Monetary)
features from the cleaned retail dataset.

Input:
    Data/cleaned_retail.csv

Output:
    Data/rfm_data.csv
"""

import os
import pandas as pd

# ==========================================
# FILE PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLEAN_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "cleaned_retail.csv"
)

RFM_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "rfm_data.csv"
)


# ==========================================
# LOAD CLEANED DATA
# ==========================================

def load_data(path=CLEAN_PATH):
    """Load cleaned retail dataset."""
    df = pd.read_csv(path)
    return df


# ==========================================
# CREATE RFM DATASET
# ==========================================

def create_rfm(df):

    df = df.copy()

    # Convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Latest purchase date + 1 day
    snapshot_date = (
        df["InvoiceDate"].max()
        + pd.Timedelta(days=1)
    )

    # Create RFM table
    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=(
                "InvoiceDate",
                lambda x: (
                    snapshot_date - x.max()
                ).days
            ),

            Frequency=(
                "InvoiceNo",
                "nunique"
            ),

            Monetary=(
                "TotalPrice",
                "sum"
            )
        )
        .reset_index()
    )

    # Clean formatting
    rfm["CustomerID"] = rfm["CustomerID"].astype(int)
    rfm["Monetary"] = rfm["Monetary"].round(2)

    return rfm


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("=" * 50)
    print("SHOPPER SPECTRUM - RFM ANALYSIS")
    print("=" * 50)

    print("\nLoading cleaned dataset...")
    df = load_data()

    print(f"Dataset Shape : {df.shape}")

    print("\nCreating RFM dataset...")
    rfm = create_rfm(df)

    print("\nRFM Dataset Preview:")
    print(rfm.head())

    print("\nRFM Dataset Information:")
    print(rfm.info())

    print("\nRFM Statistics:")
    print(rfm.describe())

    print(f"\nTotal Customers : {rfm.shape[0]:,}")

    # Save file
    rfm.to_csv(
        RFM_PATH,
        index=False
    )

    print(f"\nRFM dataset saved successfully!")
    print(f"Location : {RFM_PATH}")


# ==========================================
# RUN PROGRAM
# ==========================================

if __name__ == "__main__":
    main()
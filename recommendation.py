import os
import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "cleaned_retail.csv"
)

SIMILARITY_PATH = os.path.join(
    BASE_DIR,
    "Models",
    "similarity.pkl"
)

PRODUCTS_PATH = os.path.join(
    BASE_DIR,
    "Models",
    "products.pkl"
)

def load_data():
    return pd.read_csv(DATA_PATH)

def create_product_matrix(df):

    matrix = (
        df.pivot_table(
            index="CustomerID",
            columns="Description",
            values="Quantity",
            aggfunc="sum",
            fill_value=0
        )
    )

    return matrix

def create_similarity(matrix):

    similarity = cosine_similarity(
        matrix.T
    )

    similarity_df = pd.DataFrame(
        similarity,
        index=matrix.columns,
        columns=matrix.columns
    )

    return similarity_df

def save_models(similarity_df):
    """Save similarity matrix and product list."""

    joblib.dump(
        similarity_df,
        SIMILARITY_PATH
    )

    joblib.dump(
        similarity_df.columns.tolist(),
        PRODUCTS_PATH
    )

def recommend_products(
    product_name,
    similarity_df,
    top_n=5
):

    if product_name not in similarity_df.columns:
        return f"{product_name} not found."

    recommendations = (
        similarity_df[product_name]
        .sort_values(ascending=False)
        .iloc[1:top_n + 1]
    )

    return recommendations

def main():

    print("=" * 55)
    print("SHOPPER SPECTRUM - PRODUCT RECOMMENDATION")
    print("=" * 55)

    df = load_data()

    print(
        f"\nDataset Shape: {df.shape}"
    )

    matrix = create_product_matrix(df)

    print(
        f"\nMatrix Shape: {matrix.shape}"
    )

    similarity_df = (
        create_similarity(matrix)
    )

    save_models(similarity_df)

    print(
        f"\nSaved: {SIMILARITY_PATH}"
    )

    print(
        f"Saved: {PRODUCTS_PATH}"
    )

    sample_product = (
        similarity_df.columns[0]
    )

    print(
        f"\nSample Product:"
    )

    print(sample_product)

    print(
        "\nRecommendations:"
    )

    print(
        recommend_products(
            sample_product,
            similarity_df
        )
    )

if __name__ == "__main__":
    main()
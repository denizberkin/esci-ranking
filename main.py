import pandas as pd
import os

from utils.load import load_df


# XXX: not sure to be using smooth-ed labels or not
SCORING = {"E": 3.0,  # exact
           "S": 2.0,  # substitute
           "C": 1.0,  # complementary
           "I": 0.0}  # irrelevant


def main(**kwargs):
    """ input: paths to  dfs """
    df_examples = load_df(kwargs["ex"], "parquet")
    df_products = load_df(kwargs["pr"], "parquet")
    # df_sources = load_df(kwargs["src"], "csv")

    # print(df_examples.head())
    # print(df_products.head())
    # print(df_sources.head())
    
    # join examples and products on product_id
    df_joined = pd.merge(df_examples,
                         df_products,
                         how="left",
                         left_on=["product_locale", "product_id"],
                         right_on=["product_locale", "product_id"])
    
    # print(df_joined.columns)

    print(df_joined.columns)
    df_joined_small = df_joined[df_joined["small_version"] == 1]
    distil = df_joined_small[["esci_label"]]

    # print(distil[distil["query_id"] == 2])

    print(distil.value_counts())


if __name__ == "__main__":
    ROOT = os.path.join(os.getcwd(), "shopping_queries")

    examples = os.path.join(ROOT, "dataset_examples.parquet")
    products = os.path.join(ROOT, "dataset_products.parquet")
    sources = os.path.join(ROOT, "dataset_sources.csv")

    main(ex=examples, pr=products, src=sources)
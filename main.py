import pandas as pd
import os

from utils.load import load_df


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
    
    print(df_joined.columns)



if __name__ == "__main__":
    ROOT = os.path.join(os.getcwd(), "shopping_queries")

    examples = os.path.join(ROOT, "dataset_examples.parquet")
    products = os.path.join(ROOT, "dataset_products.parquet")
    sources = os.path.join(ROOT, "dataset_sources.csv")

    main(ex=examples, pr=products, src=sources)
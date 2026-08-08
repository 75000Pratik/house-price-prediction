from sklearn.datasets import fetch_california_housing
import pandas as pd


def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.shape)

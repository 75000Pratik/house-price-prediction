from sklearn.datasets import fetch_california_housing
import pandas as pd


def load_data():
    """
    Load the California housing dataset 
    and return it as a Pandas DataFrame.
    """

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df


if __name__ == "__main__":
    df = load_data()

    print("=" * 60)
    print("California Housing Dataset")
    print("=" * 60)

    print("\nFirst 5 Rows")
    print(df.head())

    print("\nLast 5 Rows")
    print(df.tail())

    print("\nDataset Shape")
    print(df.shape)

    print("\nColumn Names")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing values")
    print(df.isnull().sum())

    print("\nSummary Statistics")
    print(df.describe())

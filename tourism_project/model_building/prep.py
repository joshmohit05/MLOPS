"""
Data preparation step.

Loads the registered dataset, drops columns that carry no predictive
signal, fixes a couple of known category typos, and splits the data into
train/test sets that get saved to the working directory (so the GitHub
Actions workflow can pass them to the training job as artifacts).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"

# Not predictive: a row index column and the customer's unique ID
DROP_COLS = ["Unnamed: 0", "CustomerID"]
TARGET = "ProdTaken"


def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known inconsistent category labels in the raw data."""
    df = df.copy()
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
    return df


def prepare_data(path: str = DATA_PATH, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(path)

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = clean_categories(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print(f"Train shape: {Xtrain.shape}   Test shape: {Xtest.shape}")
    print(f"Train target rate: {ytrain.mean():.3f}   Test target rate: {ytest.mean():.3f}")

    return Xtrain, Xtest, ytrain, ytest


if __name__ == "__main__":
    prepare_data()

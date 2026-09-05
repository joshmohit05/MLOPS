"""
Registers the raw tourism dataset: checks it exists, checks the expected
columns are present, and prints a short summary so the workflow log shows
what data the pipeline is about to train on.
"""

import os
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def register_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    print("Dataset registered successfully.")
    print(f"Path        : {path}")
    print(f"Shape       : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Target rate : ProdTaken=1 -> {df['ProdTaken'].mean():.3f}")
    print(f"Null count  : {int(df.isnull().sum().sum())} total nulls")

    return df


if __name__ == "__main__":
    register_dataset()

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "Age": [22, np.nan, 38, 26, 35],
        "Fare": [7.25, 71.28, 7.92, 53.1, 8.05],
        "Embarked": ["S", "C", np.nan, "S", "S"]
    })

def test_impute_missing_age(sample_data):
    df = sample_data.copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    assert not df["Age"].isnull().any()
    assert df["Age"].iloc[1] == 30.5

def test_impute_missing_embarked(sample_data):
    df = sample_data.copy()
    df["Embarked"] = df["Embarked"].fillna("S")
    assert not df["Embarked"].isnull().any()

def test_handle_outliers(sample_data):
    df = sample_data.copy()
    df.loc[1, "Fare"] = 500.0
    df["Fare"] = df["Fare"].clip(upper=100.0)
    assert df["Fare"].max() <= 100.0

def test_convert_types(sample_data):
    df = sample_data.copy()
    df["Age"] = df["Age"].fillna(29.5).astype(int)
    assert df["Age"].dtype == np.int32 or df["Age"].dtype == np.int64

def test_remove_duplicates():
    df = pd.DataFrame({"A": [1, 1, 2], "B": [3, 3, 4]})
    df = df.drop_duplicates()
    assert len(df) == 2

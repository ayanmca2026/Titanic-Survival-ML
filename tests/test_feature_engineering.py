import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def feature_data():
    return pd.DataFrame({
        "SibSp": [1, 0, 3, 0],
        "Parch": [0, 0, 1, 2],
        "Name": ["Braund, Mr. Owen Harris", "Cumings, Mrs. John Bradley", "Heikkinen, Miss. Laina", "Panula, Master. Juha Niilo"],
        "Age": [22, 38, 26, 2]
    })

def test_family_size(feature_data):
    df = feature_data.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    assert list(df["FamilySize"]) == [2, 1, 5, 3]

def test_is_alone(feature_data):
    df = feature_data.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    assert list(df["IsAlone"]) == [0, 1, 0, 0]

def test_title_extraction(feature_data):
    df = feature_data.copy()
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    assert list(df["Title"]) == ["Mr", "Mrs", "Miss", "Master"]

def test_age_group(feature_data):
    df = feature_data.copy()
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 60, 100], labels=["Child", "Adult", "Senior"])
    assert list(df["AgeGroup"]) == ["Adult", "Adult", "Adult", "Child"]

def test_encoding(feature_data):
    df = feature_data.copy()
    df["Sex"] = ["male", "female", "female", "male"]
    df = pd.get_dummies(df, columns=["Sex"], drop_first=True)
    assert "Sex_male" in df.columns
    assert list(df["Sex_male"]) == [1, 0, 0, 1]

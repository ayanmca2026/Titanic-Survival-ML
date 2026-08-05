import pytest
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import os

@pytest.fixture
def train_data():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y

def test_model_training(train_data):
    X, y = train_data
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)
    assert hasattr(model, "classes_")

def test_model_saving_loading(train_data, tmpdir):
    X, y = train_data
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)
    filepath = os.path.join(tmpdir, "model.joblib")
    joblib.dump(model, filepath)
    loaded_model = joblib.load(filepath)
    preds = loaded_model.predict(X)
    assert len(preds) == 100

def test_prediction_output_format(train_data):
    X, y = train_data
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})

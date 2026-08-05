import pytest
import numpy as np

def dummy_predict(features):
    return 1

def dummy_predict_batch(features_list):
    return [1, 0]

def test_single_prediction():
    pred = dummy_predict([1, 0, 25.0, 1, 0, 50.0])
    assert pred in [0, 1]

def test_batch_prediction():
    preds = dummy_predict_batch([[1, 0, 25.0, 1, 0, 50.0], [3, 1, 30.0, 0, 0, 7.5]])
    assert len(preds) == 2
    assert all(p in [0, 1] for p in preds)

def test_prediction_output_schema():
    preds = dummy_predict_batch([[1, 0, 25.0, 1, 0, 50.0]])
    assert isinstance(preds, list)
    assert isinstance(preds[0], int)

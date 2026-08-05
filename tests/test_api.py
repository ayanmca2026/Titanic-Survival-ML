import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float
    SibSp: int
    Parch: int

@app.get("/")
def read_root(): return {"message": "Titanic API"}

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/predict")
def predict(passenger: Passenger): return {"survived": 1, "probability": 0.8}

@app.post("/predict_batch")
def predict_batch(passengers: List[Passenger]): return {"predictions": [{"survived": 1, "probability": 0.8} for _ in passengers]}

client = TestClient(app)

def test_get_root():
    response = client.get("/")
    assert response.status_code == 200

def test_get_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_post_predict_valid():
    payload = {"Pclass": 1, "Sex": "female", "Age": 25.0, "Fare": 50.0, "SibSp": 0, "Parch": 0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "survived" in response.json()

def test_post_predict_invalid():
    payload = {"Pclass": "First", "Sex": "female"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_post_predict_batch():
    payload = [
        {"Pclass": 1, "Sex": "female", "Age": 25.0, "Fare": 50.0, "SibSp": 0, "Parch": 0},
        {"Pclass": 3, "Sex": "male", "Age": 30.0, "Fare": 7.5, "SibSp": 0, "Parch": 0}
    ]
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2

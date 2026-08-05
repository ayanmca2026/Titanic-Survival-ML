from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class PassengerInput(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Ticket class (1=1st, 2=2nd, 3=3rd)")
    Sex: str = Field(..., description="Sex (male/female)")
    Age: float = Field(..., description="Age in years")
    SibSp: int = Field(..., description="Number of siblings / spouses aboard")
    Parch: int = Field(..., description="Number of parents / children aboard")
    Fare: float = Field(..., description="Passenger fare")
    Embarked: str = Field(..., description="Port of Embarkation (C/Q/S)")
    Name: Optional[str] = Field(None, description="Passenger name")
    Ticket: Optional[str] = Field(None, description="Ticket number")
    Cabin: Optional[str] = Field(None, description="Cabin number")

class PredictionOutput(BaseModel):
    survived: bool = Field(..., description="Prediction: True if survived, False otherwise")
    survival_probability: float = Field(..., description="Probability of survival (0.0 to 1.0)")
    prediction_label: str = Field(..., description="String label ('Survived'/'Did Not Survive')")
    passenger_summary: Dict[str, Any] = Field(..., description="Summary of passenger inputs")
    feature_contributions: Optional[Dict[str, float]] = Field(None, description="Feature contributions to prediction (e.g., from SHAP)")

class BatchInput(BaseModel):
    passengers: List[PassengerInput] = Field(..., description="List of passengers to predict")

class BatchOutput(BaseModel):
    predictions: List[PredictionOutput] = Field(..., description="List of predictions")
    summary: Dict[str, Any] = Field(..., description="Summary statistics of the batch predictions")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    uptime: float = Field(..., description="API uptime in seconds")
    model_version: str = Field(..., description="Version of the loaded model")
    model_loaded: bool = Field(..., description="Whether the model is successfully loaded")

class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., description="Name of the model")
    version: str = Field(..., description="Model version")
    features: List[str] = Field(..., description="List of features expected by the model")
    metrics: Dict[str, Any] = Field(..., description="Model evaluation metrics")
    training_date: str = Field(..., description="Date when the model was trained")

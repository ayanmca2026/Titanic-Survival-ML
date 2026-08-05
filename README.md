# 🚢 Titanic Survival Prediction

![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CI Status](https://img.shields.io/badge/build-passing-brightgreen)

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Dashboard](#dashboard)
- [Docker Deployment](#docker-deployment)
- [CI/CD](#cicd)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Description
An end-to-end Machine Learning pipeline to predict survival on the Titanic. Includes data preprocessing, model training, a FastAPI REST API, and a Streamlit dashboard.

## Features
- Robust data preprocessing & feature engineering
- Multiple ML models (Random Forest, XGBoost, LightGBM)
- RESTful API for real-time predictions
- Interactive web dashboard for EDA and predictions
- Comprehensive test suite
- Dockerized deployment

## Tech Stack
- **Data Science:** Pandas, NumPy, Scikit-learn, XGBoost, LightGBM
- **API:** FastAPI, Uvicorn, Pydantic
- **Dashboard:** Streamlit, Plotly
- **DevOps:** Docker, GitHub Actions, Pytest

## Installation
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
- **Full pipeline:** `python run.py --mode full`
- **API:** `python run.py --mode api`
- **Dashboard:** `python run.py --mode dashboard`

## Project Structure
```
Titanic-Survival-Prediction/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── api/
│   └── dashboard/
├── tests/
├── .github/workflows/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Model Performance
| Model | Accuracy | F1-Score |
|---|---|---|
| Random Forest | 0.82 | 0.81 |
| XGBoost | 0.84 | 0.83 |
| LightGBM | 0.83 | 0.82 |

## API Documentation
- `GET /`: API Root
- `GET /health`: Health check
- `POST /predict`: Predict for a single passenger
- `POST /predict_batch`: Predict for multiple passengers

## Dashboard
*Screenshots placeholder*

## Docker Deployment
```bash
docker-compose up --build
```
API runs on port 8000. Dashboard runs on port 8501.

## CI/CD
Automated testing and Docker image building via GitHub Actions on pushes to `main`.

## Contributing
1. Fork the repo
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License
MIT License. See [LICENSE](LICENSE) for details.

## Contact
Titanic ML Team - titanic-ml@example.com

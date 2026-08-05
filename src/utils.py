import os
import logging
import json
import joblib

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

ensure_dir(MODELS_DIR)
ensure_dir(IMAGES_DIR)
ensure_dir(REPORTS_DIR)

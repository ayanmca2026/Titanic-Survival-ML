import os
import pandas as pd
import logging
from src.utils import DATA_URL, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def load_data(url=DATA_URL, save_local=True):
    """Load Titanic dataset from local cache or download from URL."""
    local_path = os.path.join(RAW_DATA_DIR, 'titanic.csv')
    
    # Try local first
    if os.path.exists(local_path):
        logger.info(f"Loading dataset from local cache: {local_path}")
        df = pd.read_csv(local_path)
        logger.info(f"Dataset loaded with shape {df.shape}")
        return df
    
    # Download
    logger.info(f"Downloading dataset from {url}")
    try:
        df = pd.read_csv(url)
        logger.info(f"Dataset downloaded with shape {df.shape}")
        
        if save_local:
            df.to_csv(local_path, index=False)
            logger.info(f"Dataset saved to {local_path}")
            
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def dataset_overview(df):
    """Print comprehensive dataset overview."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"\nShape: {df.shape}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nMissing Percentage:\n{(df.isnull().sum() / len(df) * 100).round(2)}")
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    print(f"\nStatistical Summary:\n{df.describe()}")
    print("=" * 60)
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': int(df.duplicated().sum()),
    }


if __name__ == "__main__":
    df = load_data()
    dataset_overview(df)
    print(df.head())

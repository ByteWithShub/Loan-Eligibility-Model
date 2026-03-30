#Data loading module for the loan eligibility model. 
#This module contains a function to load data from a CSV file and return it as a pandas DataFrame. 
#It also includes error handling to log any issues that occur during the data loading process.

import pandas as pd
from src.logger import setup_logger

logger = setup_logger()


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
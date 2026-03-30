#Prediction module for the loan eligibility model. 
#This module contains a function to make predictions using a trained model.


import pandas as pd
from src.logger import setup_logger

logger = setup_logger()


def make_prediction(model, input_data):
    """
    Predict loan approval from input data.
    """
    try:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        logger.info("Prediction made successfully")
        return prediction
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise
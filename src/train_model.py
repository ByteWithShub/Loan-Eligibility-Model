#Model training module for the loan eligibility model. 
#This module contains functions to train different machine learning models and save them.


import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.logger import setup_logger

logger = setup_logger()


def train_logistic_regression(xtrain, ytrain):
    model = LogisticRegression()
    model.fit(xtrain, ytrain)
    logger.info("Logistic Regression model trained")
    return model


def train_decision_tree(xtrain, ytrain):
    model = DecisionTreeClassifier()
    model.fit(xtrain, ytrain)
    logger.info("Decision Tree model trained")
    return model


def train_random_forest(xtrain, ytrain):
    model = RandomForestClassifier()
    model.fit(xtrain, ytrain)
    logger.info("Random Forest model trained")
    return model


def save_model(model, file_path: str):
    try:
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, file_path)
        logger.info(f"Model saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise
#Data preprocessing module for the loan eligibility model. 
#This module contains functions for cleaning, encoding, and scaling the data.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.logger import setup_logger

logger = setup_logger()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and basic cleaning.
    """
    try:
        df = df.copy()

        df["Credit_History"] = df["Credit_History"].astype("object")
        df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype("object")

        df["Gender"].fillna("Male", inplace=True)
        df["Married"].fillna(df["Married"].mode()[0], inplace=True)
        df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
        df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
        df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)
        df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

        df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)

        logger.info("Missing values handled successfully")
        return df

    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise


def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables and prepare target.
    """
    try:
        df = df.copy()

        if "Loan_ID" in df.columns:
            df = df.drop("Loan_ID", axis=1)

        df = pd.get_dummies(
            df,
            columns=[
                "Gender",
                "Married",
                "Dependents",
                "Education",
                "Self_Employed",
                "Property_Area"
            ],
            dtype=int
        )

        df["Loan_Approved"] = df["Loan_Approved"].replace({"Y": 1, "N": 0})

        logger.info("Categorical encoding completed")
        return df

    except Exception as e:
        logger.error(f"Error during encoding: {e}")
        raise


def split_features_target(df: pd.DataFrame):
    """
    Split dataframe into features (X) and target (y).
    """
    try:
        X = df.drop("Loan_Approved", axis=1)
        y = df["Loan_Approved"]
        logger.info("Features and target separated")
        return X, y
    except Exception as e:
        logger.error(f"Error splitting features and target: {e}")
        raise


def scale_data(xtrain, xtest):
    """
    Scale data using MinMaxScaler.
    """
    try:
        scaler = MinMaxScaler()
        xtrain_scaled = scaler.fit_transform(xtrain)
        xtest_scaled = scaler.transform(xtest)
        logger.info("Data scaling completed")
        return xtrain_scaled, xtest_scaled, scaler
    except Exception as e:
        logger.error(f"Error during scaling: {e}")
        raise
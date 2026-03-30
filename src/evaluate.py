#Model evaluation module for the loan eligibility model. 
#This module contains functions to evaluate trained models and perform cross-validation.


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold
from src.logger import setup_logger

logger = setup_logger()


def evaluate_model(model, xtest, ytest):
    try:
        ypred = model.predict(xtest)
        accuracy = accuracy_score(ytest, ypred)
        cm = confusion_matrix(ytest, ypred)

        logger.info(f"Model evaluated successfully with accuracy: {accuracy:.4f}")
        return accuracy, cm
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


def cross_validate_model(model, X, y, folds=5):
    try:
        kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kfold)
        logger.info("Cross-validation completed")
        return scores
    except Exception as e:
        logger.error(f"Error during cross-validation: {e}")
        raise
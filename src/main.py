#Main execution script for the loan eligibility model.

from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import clean_data, encode_data, split_features_target, scale_data
from src.train_model import train_logistic_regression, save_model
from src.evaluate import evaluate_model, cross_validate_model
import joblib

def main():
    df = load_data("credit.csv")
    df = clean_data(df)
    df = encode_data(df)

    X, y = split_features_target(df)

    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    xtrain_scaled, xtest_scaled, scaler = scale_data(xtrain, xtest)

    model = train_logistic_regression(xtrain_scaled, ytrain)

    accuracy, cm = evaluate_model(model, xtest_scaled, ytest)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)

    scores = cross_validate_model(model, xtrain_scaled, ytrain)
    print("Cross-validation scores:", scores)
    print("Mean CV accuracy:", scores.mean())

    save_model(model, "models/loan_model.pkl")
    joblib.dump(xtrain.columns.tolist(), "models/columns.pkl")

if __name__ == "__main__":
    main()
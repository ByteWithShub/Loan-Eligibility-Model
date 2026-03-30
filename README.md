## Loan Eligibility Prediction System

> “Good decisions come from good data, great systems make those decisions consistent.”

---

### Overview

This project focuses on building a machine learning system that predicts whether a loan application is likely to be approved based on applicant details. 

Originally developed as a Jupyter Notebook, the solution has been fully **modularized into a production-style Python project**, deployed using **Streamlit**, and structured for scalability, readability, and real-world use.

The goal is not just prediction but building a system that reflects how ML solutions are designed and deployed in industry.

---

### Objectives

- Predict loan approval status using classification models
- Achieve accuracy greater than 76%
- Transform notebook code into a modular, maintainable project
- Build an interactive web application for real-time predictions
- Follow best practices for ML deployment

---

### Machine Learning Approach

- **Task Type:** Classification  
- **Target Variable:** `Loan_Approved`  
- **Models Used:**
  - Logistic Regression
  - Decision Tree
  - Random Forest  

- **Evaluation Metrics:**
  - Accuracy Score
  - Confusion Matrix
  - Cross Validation

---

### Features

- Data preprocessing (missing value handling, encoding)
- Feature scaling using MinMaxScaler
- Modular code structure (separated into reusable components)
- Model training and evaluation pipeline
- Logging and error handling for robustness
- Interactive Streamlit web app
- GitHub-ready project structure

---

### Project Structure
```
loan_eligibility_project/
│
├── app.py # Streamlit app
├── main.py # Training pipeline
├── requirements.txt # Dependencies
├── README.md # Project documentation
├── credit.csv # Dataset
│
├── models/ # Saved models & columns
├── logs/ # Log files
│
└── src/
├── data_loader.py
├── preprocessing.py
├── train_model.py
├── evaluate.py
├── predict.py
└── logger.py
```


---

### Installation & Setup
```
1. Clone the repository:
```bash
git clone <your-repo-link>
cd loan_eligibility_project

2. Install Dependencies
pip install -r requirements.txt

3. Train the model
python main.py

4. Run Streamlit 
streamlit run app.py
```

### How It Works
User inputs applicant details
Data is preprocessed (same pipeline as training)
Model predicts loan approval
Output shows:
Approval status
Prediction confidence

### Technologies Used
Python
Pandas & NumPy
Scikit-learn
Streamlit
Joblib

### Model Performance
Achieved accuracy above required threshold (76%)
Cross-validation used for reliable performance estimation
Logistic Regression selected as baseline model

### Key Learning Outcomes
Transitioning from notebook to production-level code
Importance of modular design in ML systems
Handling real-world issues like missing data and feature consistency
Deploying ML models as interactive applications


## Author
```
Shubhangi Singh 
``` 

# Future Improvements
Hyperparameter tuning for improved accuracy
Model comparison dashboard
Feature importance visualization
Deployment with cloud APIs

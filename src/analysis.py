import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    average_precision_score, 
    auc
)
from sklearn.model_selection import GridSearchCV

def main():
    # Read Data
    data = pd.read_csv("data/raw/IBM-Telco-Customer-Churn.csv")
    
    # Split Data
    s = data['TotalCharges']
    index = 0
    for c in s:

        if c == ' ':
            data['TotalCharges'][index] = None
    
        index +=1

    data['TotalCharges'] = data['TotalCharges'].astype(float)

    train_df, test_df = train_test_split(data, test_size=0.3, random_state=1)

    X_train = train_df.drop(columns=["Churn"])
    X_test = test_df.drop(columns=["Churn"])

    y_train = train_df["Churn"]
    y_test = test_df["Churn"]

    # Build Preprocessor
    numeric_features = ['MonthlyCharges', 'tenure', 'TotalCharges']

    categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
                        'PaymentMethod']

    pass_through_features = ['SeniorCitizen', 'Churn']

    preprocessor = build_preprocessor(numeric_features, categorical_features, pass_through_features)

    # Hyperparameter Optimization
    scoring = [
    "f1",
    "recall",
    "precision"
    ]

    lr_pipe = make_pipeline(preprocessor, LogisticRegression())

    param_grid = {
    "logisticregression__C": 10.0 ** np.arange(-4, 6, 1)
    }

    grid_search = GridSearchCV(
        lr_pipe,
        param_grid,
        cv=3,
        scoring=scoring,
        refit="f1"
)


def build_preprocessor(numeric_features, categorical_features, pass_through_features):
    
    # Numeric pipeline
    numeric_pipeline = Pipeline(steps=[
        ("simpleimputer", SimpleImputer()),
        ("standardscaler", StandardScaler() )
        ]
    )

    # Build preprocessor
    preprocessor = make_column_transformer(
        (numeric_pipeline, numeric_features),
        (OneHotEncoder(handle_unknown="ignore", sparse=True), categorical_features),
        ('passthrough', pass_through_features)
    )

if __name__ == "__main__":
    
    main()
# author: GROUP 12
# date: 2021-11-25
	
'''This script downloads a data file in csv format. 
This script takes an unquoted data file path to a csv file, 
the name of the file type to write the file to (ex. csv), 
and the name of a file path to write locally (including the name of the file).
	
Usage: analysis.py --train_path=<train_path> --test_path=<test_path> --out_dir=<out_dir>
	
Options:
--train_path=<train_path>   Path to the training data file (must be in standard csv format)
--test_path=<test_path>    Path to the test data file (must be in standard csv format)
--out_dir=<out_dir>    Output directory path where the result figures will be saved
'''

import os
import sys
import pandas as pd
import numpy as np
from docopt import docopt
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV

opt = docopt(__doc__)
class_report_name = "classification_report.csv"
confusion_matrix_name = "confusion_matrix.png"

def main(train_path, test_path, out_dir):

    # Read train & test data
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except Exception as e:
        print("Unable to read training/test data. Please check filepath's.")
    
    # Split into features/targets
    X_train = train_df.drop(columns=["Churn"])
    print(X_train.head())

    X_test = test_df.drop(columns=["Churn"])

    y_train = train_df["Churn"]
    y_test = test_df["Churn"]

    # Build Preprocessor
    numeric_features = ['MonthlyCharges', 'tenure', 'TotalCharges']

    categorical_features = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
                        'PaymentMethod', 'SeniorCitizen']

    drop_features = ["customerID", "gender"]

    preprocessor = build_preprocessor(numeric_features, categorical_features, drop_features)

    # Build pipeline for hyperparameter optimization
    lr_pipe = make_pipeline(preprocessor, LogisticRegression(max_iter=10000))

    param_grid = {
        "logisticregression__C": 10.0 ** np.arange(-4, 6, 1),
        "logisticregression__class_weight":[None, "balanced"]
    }

    # We're concerned with finding parameters that optimize f1 score
    grid_search = GridSearchCV(
        lr_pipe,
        param_grid,
        cv=4,
        scoring="f1"
    )

    # Get best parameters from grid_search
    best_coefs = grid_search.fit(X_train, y_train).best_params_

    # Define best model pipeline
    best_lr_pipe = make_pipeline(preprocessor, 
                                LogisticRegression(C=best_coefs["logisticregression__C"], 
                                                   class_weight=best_coefs["logisticregression__class_weight"]))

    # Fit best model on training data
    best_lr_pipe.fit(X_train, y_train)

    # Grab learned coefficients from model. We'll need these to report on which features are most 
    # meaningful for predicting churn. 
    feats = best_lr_pipe.named_steps["logisticregression"].coef_[0]

    # Build list of column names. There are some new columns from one hot encoder. 
    cols = (numeric_features + 
        list(preprocessor.named_transformers_["onehotencoder"].get_feature_names_out()))


    # Build & export dataframe of most positively & negatively correlated features
    #TODO

    # Run best model on test data
    preds = best_lr_pipe.predict(X_test)

    # Generate classification report
    class_report = classification_report(
        y_test, preds, target_names=["non-Churn", "Churn"], output_dict=True
    )
    class_report_df = pd.DataFrame(class_report).transpose()
    print(class_report_df)
    try:
        class_report_df.to_csv(os.path.join(out_dir, class_report_name))
    except:
        os.makedirs(os.path.dirname(out_dir))
        class_report_df.to_csv(os.path.join(out_dir, class_report_name))


def build_preprocessor(numeric_features, 
                        categorical_features, 
                        drop_features=[]):
    
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
        ("drop", drop_features)
    )

    return preprocessor

if __name__ == "__main__":
    
    # Have the user input train data path, test data path, and output directory
    main(opt["--train_path"], opt["--test_path"], opt["--out_dir"])
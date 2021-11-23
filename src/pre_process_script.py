
# author: Adam Morphy
# date: 2021-11-23
	
'''
Cleans, splits and pre-processes (scales, encodes and imputes) the Telco Churn data (from hhttps://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv).
Writes the training and test data to separate csv files.

Usage: src/pre_process_script.py --input=<input> --out_dir=<out_dir>
  
Options:
--input=<input>       Path (including filename) to raw data (csv file)
--out_dir=<out_dir>   Path to directory where the processed data should be written
'''
	
import pandas as pd
import numpy as np
from docopt import docopt
import os
import requests
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
	
opt = docopt(__doc__)
	
def main(input, out_dir):

    # read data and convert class to pandas df
    raw_data = pd.read_csv(input) 

    # Converting TotalCharges from object to float (pd functions astype(float))
    index = 0
    s = raw_data['TotalCharges']

    for c in s:

        if c == ' ':
            raw_data['TotalCharges'][index] = None
    
        index +=1

    raw_data['TotalCharges'] = raw_data['TotalCharges'].astype(float)  


    # split into training and test data sets
    train_df, test_df = train_test_split(raw_data, test_size=0.3, random_state=1)


    # Define column transformer for transformations
    numeric_features = ['MonthlyCharges', 'tenure', 'TotalCharges']

    categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
                        'PaymentMethod']

    pass_through_features = ['SeniorCitizen', 'Churn']

    numeric_pipeline = Pipeline(steps=[
        ("simpleimputer", SimpleImputer()),
        ("standardscaler", StandardScaler() )
        ]
    )

    preprocessor = make_column_transformer(
                                            (numeric_pipeline, numeric_features),
                                            (OneHotEncoder(handle_unknown="ignore", sparse=True), categorical_features),
                                            ('passthrough', pass_through_features)
                                        )


    transformed_train = preprocessor.fit_transform(train_df)

    transformed_test = preprocessor.fit_transform(test_df)

    #list(preprocessor.named_transformers_['pipeline']['onehotencoder'].get_feature_names_out())

    transformed_train = pd.DataFrame(transformed_train,
                                        columns = (numeric_features + list(preprocessor.named_transformers_['onehotencoder'].get_feature_names_out()) + pass_through_features)
                                        ).head(5)


    transformed_test = pd.DataFrame(transformed_test,
                                        columns = (numeric_features + list(preprocessor.named_transformers_['onehotencoder'].get_feature_names_out()) + pass_through_features)
                                        ).head(5)



  
    # write training and test data to csv files
    train_df.to_csv((out_dir + 'train_df.csv'), index = False)
    test_df.to_csv((out_dir + 'test_df.csv'), index = False)

    print("Data successfully stored in: ", (out_dir + 'train_df.csv'), " and ", (out_dir + 'train_df.csv'))


if __name__ == "__main__":
    
    # Call main method, and have the user input the raw file, out dir
    main(opt["--input"], opt["--out_dir"])
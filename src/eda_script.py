# author: Anupriya Srivastava
# date: 2021-11-24

'''
Performs exploratory data analysis on the Telco Churn data (from hhttps://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv). Saves output figures in .png files.

Usage: src/eda_script.py --input=<input> --out_dir=<out_dir>
  
Options:
--input=<input>       Path (including filename) to cleaned data (csv file)
--out_dir=<out_dir>   Path to directory where the figures should be saved
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

alt.renderers.enable('mimetype')

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

    # Change target variable to a boolean
    raw_data['Churn'] = raw_data['Churn'].replace("Yes", True)
    raw_data['Churn'] = raw_data['Churn'].replace("No", False)

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
                                        )


    transformed_test = pd.DataFrame(transformed_test,
                                        columns = (numeric_features + list(preprocessor.named_transformers_['onehotencoder'].get_feature_names_out()) + pass_through_features)
                                        )




    # write training and test data to csv files
    transformed_train.to_csv((out_dir + 'train_df.csv'), index = False)
    transformed_test.to_csv((out_dir + 'test_df.csv'), index = False)

    print("Data successfully stored in: ", (out_dir + 'train_df.csv'), " and ", (out_dir + 'train_df.csv'))


if __name__ == "__main__":

    # Call main method, and have the user input the raw file, out dir
    main(opt["--input"], opt["--out_dir"])

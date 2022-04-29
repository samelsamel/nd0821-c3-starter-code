# Script to train machine learning model.
'''
Author: Amel Sellami 
Date: 28-04-2022
Goal: loading data, run the train command and save the model.
'''
import joblib
import numpy as np
import os
import pandas as pd 
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from .ml.data import process_data
from .ml.model import train_model

# constants
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Add code to load in the data.
def load_data(data_path): 
    data = pd.read_csv(data_path)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    # this returns train_data and test_data in order.
    return train_test_split(data, test_size=0.20)

def trainer(train, model_path):

    # first process the data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    # train the model
    model = train_model(X_train, y_train)

    dirname = os.path.dirname(model_path)
    # save the full pack
    joblib.dump((model,encoder, lb), model_path)

    # save model
    joblib.dump(model, f"{dirname}/model.joblib")
    # save the encoder
    joblib.dump(encoder, f"{dirname}/encoder.joblib")
    # save the lb
    joblib.dump(lb, f"{dirname}/lb.joblib")


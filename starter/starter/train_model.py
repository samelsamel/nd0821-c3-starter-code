# Script to train machine learning model.
'''
Author: Amel Sellami
Date: 28-04-2022
Goal: loading data, run the train command and save the model.
'''
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from .ml.data import process_data
from .ml.model import train_model, compute_model_metrics, inference

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

    # Optional enhancement,
    # use K-fold cross validation instead of a train-test split.
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
    joblib.dump((model, encoder, lb), model_path)

    # save model
    joblib.dump(model, f"{dirname}/model.joblib")
    # save the encoder
    joblib.dump(encoder, f"{dirname}/encoder.joblib")
    # save the lb
    joblib.dump(lb, f"{dirname}/lb.joblib")


def batch_inference(test_data, model_path,
                    cat_features, label_column='salary'):
    # load the model from `model_path`
    model = joblib.load(open("model/model.joblib", "r+b"))
    encoder = joblib.load(open("model/encoder.joblib", 'r+b'))
    lb = joblib.load(open("model/lb.joblib", 'r+b'))

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test_data,
        categorical_features=cat_features,
        label=label_column,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # evaluate model
    preds = inference(model=model, X=X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print('Precision:\t', precision)
    print('Recall:\t', recall)
    print('F-beta score:\t', fbeta)

    return precision, recall, fbeta


def online_inference(row_dict, model_path, cat_features):
    # load the model from `model_path`
    model, encoder, lb = joblib.load(model_path)

    row_transformed = list()
    X_categorical = list()
    X_continuous = list()

    for key, value in row_dict.items():
        mod_key = key.replace('_', '-')
        if mod_key in cat_features:
            X_categorical.append(value)
        else:
            X_continuous.append(value)

    y_cat = encoder.transform([X_categorical])
    y_conts = np.asarray([X_continuous])

    row_transformed = np.concatenate([y_conts, y_cat], axis=1)

    # get inference from model
    preds = inference(model=model, X=row_transformed)

    return '>50K' if preds[0] else '<=50K'

"""
Author: Amel Sellami
Date: 28-04-2022
Goal: loading data, run the train command and save the model.
"""
import joblib
import pandas as pd
from starter.ml.model import inference, compute_model_metrics
from starter.train_model import trainer, load_data
from starter.ml.data import process_data

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
label_column = 'salary'

if __name__ == '__main__':
    data_path = 'data/census_clean.csv'
    model_path = "/home/amel/work/udacity/nd0821-c3-starter-code/model/"

    # Get the splitted data
    train_data, test_data = load_data(data_path)
    train_data.drop(['Unnamed: 0'], axis=1, inplace=True)
    test_data.drop(['Unnamed: 0'], axis=1, inplace=True)
    # Training the model on the train data
    trainer(train_data, model_path)
    # evaluating the model on the test data
    # Test model and show metrics

    model = joblib.load(open("model/model.joblib", "r+b"))
    encoder = joblib.load(open("model/encoder.joblib", 'r+b'))
    lb = joblib.load(open("model/lb.joblib", 'r+b'))

    data = pd.read_csv('data/census_clean.csv')
    # Proces the test data with the process_data function.
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    X_test, y_test, encoder, lb = process_data(
        data,
        categorical_features=CAT_FEATURES,
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

'''
Author: Amel Sellami 
Date: 28-04-2022
Goal: contains functions like train model, comute metrics and predictions.
'''
from .data import process_data
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    clf = RandomForestClassifier(random_state=1994, max_depth=19, n_estimators=64)
    clf.fit(X_train, y_train)

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def compute_slice_metrics(df,category,model,encoder,binarizer):
    """
    Computes model metrics based on data slices
    
    Inputs
    ------
    df : pd.DataFrame
         Dataframe containing the cleaned data
    category : str
         Dataframe column to slice
    rf_model: 
         Random forest model used to perform prediction
    encoder: OneHotEncoder
         Trained OneHotEncoder
    binarizer: LabelBinarizer
        Trained LabelBinarizer
     Returns
     -------
     predictions : dict
          Dictionary containing the predictions for each category feature
    """

    predictions = {}
    for cat_feature in df[category].unique():
        filtered_df = df[df[category] == cat_feature]

        X, y, _, _ = process_data(filtered_df,
                                  label='salary',
                                  training=False,
                                  encoder=encoder,
                                  lb=binarizer)

        y_preds = model.predict(X)

        precision, recall, fbeta = compute_model_metrics(y, y_preds)
        predictions[cat_feature] = {
            'precision': precision,
            'recall': recall,
            'fbeta': fbeta,
            'n_row': len(filtered_df)}
    return predictions
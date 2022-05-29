# Put the code for your API here.
'''
Author: Amel Sellami
Date: 28-04-2022
Goal: main functions for running the API call
'''
import os
import pandas as pd
import uvicorn
import joblib

from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.starter.ml.data import process_data

# FastAPI instance
app = FastAPI()

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
# DVC set-up for Heroku
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc config core.hardlink_lock true")
    if os.system("dvc pull  -r storage") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


class Input(BaseModel):
    age: int = Field(..., example=23)
    workclass: str = Field(..., example="Self-emp-inc")
    fnlgt: int = Field(..., example=76516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(..., example="Married-civ-spouse",
                                alias="marital-status")
    occupation: str = Field(..., example="Exec-managerial")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States",
                                alias="native-country")


class Output(BaseModel):
    prediction: str


@app.get("/")
def welcome():
    return "Hello World! Happy API checking"


model = joblib.load(open("model/model.joblib", "r+b"))
encoder = joblib.load(open("model/encoder.joblib", 'r+b'))
labelb = joblib.load(open("model/lb.joblib", 'r+b'))



@app.post("/predict/", response_model=Output, status_code=200)
def predict(data: Input):

    # Categorical features for transform model
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]


    # load predict_data
    request_dict = data.dict(by_alias=True)
    request_data = pd.DataFrame(request_dict, index=[0])
    data = pd.read_csv('data/census_clean.csv')
    # Proces the test data with the process_data function.
    X, _ , _, _ = process_data(
        request_data,
                categorical_features=cat_features,
                training=False,
                encoder=encoder,
                lb=labelb)
    prediction = model.predict(X)

    if prediction[0] == 1:
        prediction = "Salary > 50k"
    else:
        prediction = "Salary <= 50k"
    return {"income class": prediction}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

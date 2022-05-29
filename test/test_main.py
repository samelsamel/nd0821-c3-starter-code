'''
Author: Amel Sellami
Date: 28-04-2022
Goal: Testing the API functions
'''
import os
import inspect
import sys
from fastapi.testclient import TestClient
from starter.main import app

# Load app from parent folder:
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

client = TestClient(app)


def test_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == 'Hello World! Happy API checking'


def test_predict_greater50k():
    r = client.post("/predict/", json={
        "age": 40,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 20000,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    })

    assert r.status_code == 200
    print(r.json())
    assert r.json() == {'income class': 'Salary > 50k'}


def test_predict_less50k():
    r = client.post("/predict/", json={
        "age": 38,
        "workclass": "Private",
        "fnlgt": 215646,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Divorced",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {'income class': 'Salary <= 50k'}

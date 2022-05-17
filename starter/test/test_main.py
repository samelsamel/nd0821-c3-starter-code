'''
Author: Amel Sellami
Date: 28-04-2022
Goal: Testing the API functions
'''
import sys
from fastapi.testclient import TestClient
from starter.main import app

sys.path.append('path/tostarter/ml')

client = TestClient(app)


def test_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Hello World! Happy API checking"


def test_predict_greater50k():
    request = client.post("/", json={'age': 33,
                                     'workclass': 'Private',
                                     'fnlgt': 149184,
                                     'education': 'HS-grad',
                                     'marital_status': 'Never-married',
                                     'occupation': 'Prof-specialty',
                                     'relationship': 'Not-in-family',
                                     'race': 'White',
                                     'sex': 'Male',
                                     'hoursPerWeek': 60,
                                     'nativeCountry': 'United-States'
                                     })
    assert request.status_code == 405
    assert request.json() == {"prediction": ">50K"}


def test_predict_less50k():
    request = client.post("/", json={'age': 19,
                                     'workclass': 'Private',
                                     'fnlgt': 149184,
                                     'education': 'HS-grad',
                                     'marital_status': 'Never-married',
                                     'occupation': 'Prof-specialty',
                                     'relationship': 'Not-in-family',
                                     'race': 'White',
                                     'sex': 'Male',
                                     'hoursPerWeek': 60,
                                     'nativeCountry': 'United-States'
                                     })
    assert request.status_code == 405
    assert request.json() == {"prediction": "<=50K"}

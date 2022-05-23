'''
Author: Amel Sellami
Date: 28-04-2022
Goal: Testing the API functions
'''
import sys
from fastapi.testclient import TestClient
from starter.main import app

sys.path.append('/home/amel/work/udacity/nd0821-c3-starter-code/starter/starter/ml')

client = TestClient(app)


def test_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Hello World! Happy API checking"


# def test_predict_gr50k():
#     r = client.post("/predict/", json={
#         "age": 40,
#         "workclass": "Private",
#         "fnlgt": 193524,
#         "education": "Doctorate",
#         "education-num": 16,
#         "marital-status": "Married-civ-spouse",
#         "occupation": "Prof-specialty",
#         "relationship": "Husband",
#         "race": "White",
#         "sex": "Male",
#         "capital-gain": 15,
#         "capital-loss": 0,
#         "hours-per-week": 60,
#         "native-country": "United-States"
#     })

#     assert r.status_code == 307
#     assert r.json() == {"prediction": ">50K"}

# def test_predict_ls50k():
#     r = client.post("/predict/", json={
#         "age": 47,
#         "workclass": "Private",
#         "fnlwgt": 51835,
#         "education": "Prof-school",
#         "education_num": 15,
#         "marital_status": "Married-civ-spouse",
#         "occupation": "Prof-specialty",
#         "relationship": "Wife",
#         "race": "White",
#         "sex": "Female",
#         "capital_gain": 0,
#         "capital_loss": 1902,
#         "hours_per_week": 60,
#         "native_country": "Honduras"
#     })

#     assert r.status_code == 307
#     assert r.json() == {"prediction": "<=50K"}
'''
Author: Amel Sellami
Date: 28-04-2022
Goal: Testing the API functions
'''
from fastapi.testclient import TestClient
from starter.starter.main import app

client = TestClient(app)


def test_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Hello World! Happy API checking"

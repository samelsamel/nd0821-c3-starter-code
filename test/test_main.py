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

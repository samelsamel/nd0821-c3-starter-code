import requests
import json

test1 = {
  "age": 40,
  "workclass": "private",
  "fnlgt": 154374,
  "education": "HS-grad	",
  "education_num": 9,
  "marital-status": "Married-civ-spouse",
  "occupation": "Machine-op-inspct",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital_gain": 0,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native-country": "United-States"
}
response1 = requests.post('https://udacityappamel.herokuapp.com/predict/',
                          data=json.dumps(test1))

print(response1.status_code)
print(response1.json())

import json
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_get_root():
    r = client.get("/")
    print(r.json())
    assert r.status_code == 200

def test_get_predict():
    r = client.get("/predict")
    print(r.json())
    assert r.status_code != 200

def test_post_example_0():
    data = json.dumps({
        "age": 30,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
        })
    r = client.post("/predict", data=data)
    print(r.json())
    assert r.status_code == 200
    assert r.json() == "<=50K"


def test_post_example_1():
    data = json.dumps({
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 14084,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
        })
    r = client.post("/predict", data=data)
    print(r.json())
    assert r.status_code == 200
    assert r.json() == ">50K"

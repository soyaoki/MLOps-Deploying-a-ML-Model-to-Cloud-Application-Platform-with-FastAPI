import requests

print("Starting requests POSTing on live API")
URL = "https://mlops-deploying-a-ml-model-to-cloud.onrender.com/predict"
data = {
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
    }

print("Data requested: ", data)
respone = requests.post(URL, json=data)

print("Status code: ", respone.status_code)
print("Predict Result: ", respone.json())
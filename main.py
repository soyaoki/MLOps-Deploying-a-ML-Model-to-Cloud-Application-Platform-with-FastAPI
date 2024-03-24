from fastapi import FastAPI
from pydantic import BaseModel, Field

import pandas as pd
import pickle
import os

from ml.data import process_data
from ml.model import inference

# Instantiate the app.
app = FastAPI()


class InputData(BaseModel):
    age: int = Field(..., example=30)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    marital_status: str = Field(..., example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=40)
    native_country: str = Field(..., example="United-States")


# Define a GET on the specified endpoint.
@app.get("/")
async def welcome():
    return {"Welcome to API."}


# Define a POST on the specified endpoint.
@app.post("/predict")
async def predict(inputdata: InputData):
    data = {
        "age": inputdata.age,
        "workclass": inputdata.workclass,
        "fnlwgt": inputdata.fnlgt,
        "education": inputdata.education,
        "education-num": inputdata.education_num,
        "marital-status": inputdata.marital_status,
        "occupation": inputdata.occupation,
        "relationship": inputdata.relationship,
        "race": inputdata.race,
        "sex": inputdata.sex,
        "capital-gain": inputdata.capital_gain,
        "capital-loss": inputdata.capital_loss,
        "hours-per-week": inputdata.hours_per_week,
        "native-country": inputdata.native_country
    }

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    model_path = './model'

    model = pickle.load(open(os.path.join(model_path, 'model.pkl'), 'rb'))
    encoder = pickle.load(open(os.path.join(model_path, 'encoder.pkl'), 'rb'))
    lb = pickle.load(open(os.path.join(model_path, 'lb.pkl'), 'rb'))

    X, _, _, _ = process_data(pd.DataFrame([data]),
                              categorical_features=cat_features,
                              training=False,
                              encoder=encoder, lb=lb)

    y_pred = inference(model, X)

    return lb.inverse_transform(y_pred)[0]

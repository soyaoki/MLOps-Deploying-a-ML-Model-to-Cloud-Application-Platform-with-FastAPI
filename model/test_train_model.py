import pytest
import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from .ml.data import process_data
from .ml.model import inference

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

@pytest.fixture(scope="module")
def data():
    data_path = 'data'
    return pd.read_csv(os.path.join(data_path, 'census.csv'))

def test_data_input(data):
    """
        Test the input data
    """
    assert data.shape[0] > 0
    assert data.shape[1] > 0
    
def test_data_process(data):
    """
        Test the dataset for train and test
    """
    train, test = train_test_split(data, random_state=42, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features,  label="salary", training=False, encoder=encoder, lb=lb
    )
    assert len(X_train) + len(X_test) == len(data)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    
def test_model_output(data):
    """
        Test the model inference
    """
    model_path = 'model'
    model = pickle.load(open(os.path.join(model_path, 'model.pkl'), 'rb'))

    train, test = train_test_split(data, random_state=42, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    y_pred = inference(model, X_test)
    assert len(y_test) == len(y_pred)
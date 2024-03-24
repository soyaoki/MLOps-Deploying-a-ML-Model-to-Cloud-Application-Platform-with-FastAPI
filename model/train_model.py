# Script to train machine learning model.

from sklearn.model_selection import train_test_split

import os
import pandas as pd
import pickle

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, \
    inference, evaluate_model_performance_by_slice

# Add code to load in the data.
data_path = 'data'
data = pd.read_csv(os.path.join(data_path, 'census.csv'))

# Optional enhancement, use K-fold cross validation
# instead of a train-test split.
train, test = train_test_split(data, random_state=42, test_size=0.20)

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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
print(precision, recall, fbeta)

model_path = 'model'
pickle.dump(model, open(os.path.join(model_path, 'model.pkl'), 'wb'))
pickle.dump(encoder, open(os.path.join(model_path, 'encoder.pkl'), 'wb'))
pickle.dump(lb, open(os.path.join(model_path, 'lb.pkl'), 'wb'))

# Evaluate performances of the model on slices of the data.
evaluate_model_performance_by_slice(model, test, encoder, lb, cat_features)

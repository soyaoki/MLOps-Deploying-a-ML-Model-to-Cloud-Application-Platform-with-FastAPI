from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

from .data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ 
    Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.

    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def evaluate_model_performance_by_slice(model, data, encoder, lb, cat_features):
    """
    Evaluate the performance of a trained machine learning model on slices of the data.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    data :  pd.DataFrame
        DataFrame containing the features and label. Columns in `categorical_features`.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer passed in.
    cat_features : list[str]
        List containing the names of the categorical features (default=[]).
    
    Returns
    -------
    None
    
    """
    with open('slice_output.txt', 'w') as f:
        f.write('slice_feature'+', '+'value'+', '+'precision'+', '+'recall'+', '+'fbeta'+'\n')
        for slice_feature in cat_features:
            for value in data[slice_feature].unique():
                data_preformance = data[data[slice_feature]==value]
                X,y, _, _ = process_data(data_preformance, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
                y_pred = model.predict(X)
                precision, recall, fbeta = compute_model_metrics(y, y_pred)
                print(slice_feature, value, precision, recall, fbeta)
                f.write(str(slice_feature)+', '+str(value)+', '+str(precision)+', '+str(recall)+', '+str(fbeta)+'\n')
    
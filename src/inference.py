import joblib
import pandas as pd
import numpy as np
from src.feature_engineering import engineer_features
import yaml
import shap

def load_model(config_path):
    """Loads the trained model and mean_amount."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    try:
        model = joblib.load(config["model"]["model_path"])
        mean_amount = joblib.load(config["model"]["mean_amount_path"])
        mean_amount_by_hour = joblib.load(config["model"]["mean_amount_by_hour_path"])
        return model, mean_amount, mean_amount_by_hour
    except FileNotFoundError:
        print(f"Error: Model files not found.")
        return None, None, None

def predict(transaction_data, model, mean_amount, mean_amount_by_hour):
    """
    Predicts whether a transaction is fraudulent.

    Args:
        transaction_data: A dictionary or pandas Series representing a single transaction.
        model: The trained fraud detection model.
        mean_amount: The mean transaction amount used during training.
        mean_amount_by_hour: The mean transaction amount per hour used during training.

    Returns:
        A dictionary containing the prediction (0 or 1) and the anomaly score.
    """
    if not isinstance(transaction_data, pd.DataFrame):
        transaction_data = pd.DataFrame([transaction_data])

    transaction_data, _, _ = engineer_features(transaction_data, training_mode=False, mean_amount=mean_amount, mean_amount_by_hour=mean_amount_by_hour)

    # Ensure the order of columns matches the training data
    transaction_data = transaction_data.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(transaction_data)
    score = model.decision_function(transaction_data)  # Anomaly score

    # Convert prediction to 0 or 1 (if using Isolation Forest)
    prediction = 1 if prediction[0] == -1 else 0

    return {"prediction": prediction, "score": score[0]}

def explain_prediction_shap(transaction_data, model, mean_amount):
    """
    Explains a prediction using SHAP values.

    Args:
        transaction_data: A dictionary or pandas Series representing a single transaction.
        model: The trained fraud detection model.
        mean_amount: The mean transaction amount used during training.

    Returns:
        A SHAP explanation object.
    """

    if not isinstance(transaction_data, pd.DataFrame):
        transaction_data = pd.DataFrame([transaction_data])

    transaction_data, _, _ = engineer_features(transaction_data, training_mode=False, mean_amount=mean_amount)

    # Ensure the order of columns matches the training data
    transaction_data = transaction_data.reindex(columns=model.feature_names_in_, fill_value=0)

    explainer = shap.Explainer(model, transaction_data) # Use TreeExplainer for tree-based models if applicable
    shap_values = explainer(transaction_data)

    return shap_values
import pandas as pd
from sklearn.metrics import classification_report, average_precision_score
from sklearn.model_selection import train_test_split
import joblib
import yaml

from src.data_ingestion import load_data
from src.feature_engineering import engineer_features
from src.models import initialize_model

def train_model(config_path):
    """Trains the fraud detection model."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_data, _ = load_data(config["data"]["raw_data_path"], config_path)
    train_data, mean_amount, mean_amount_by_hour = engineer_features(train_data, training_mode=True)

    # Prepare data for model
    X_train = train_data.drop('Class', axis=1)
    y_train = train_data['Class']

    # Split a portion of training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=config["train"]["val_size"], random_state=config["train"]["random_state"]
    )

    # Model selection and training
    model = initialize_model(config)

    model.fit(X_train_split)

    # Evaluate on validation set
    if X_val is not None:
        y_val_pred = model.predict(X_val)
        if config["train"]["model"] == "IsolationForest":
            y_val_pred = [1 if x == -1 else 0 for x in y_val_pred]

        print("Validation Results:")
        print(classification_report(y_val, y_val_pred))
        print(f"Average Precision Score: {average_precision_score(y_val, y_val_pred):.4f}")

    # Retrain the model on the entire training set
    model.fit(X_train)

    # Save the trained model and mean_amount used for feature engineering
    joblib.dump(model, config["model"]["model_path"])
    joblib.dump(mean_amount, config["model"]["mean_amount_path"])
    joblib.dump(mean_amount_by_hour, config["model"]["mean_amount_by_hour_path"])

if __name__ == "__main__":
    train_model("config.yaml")
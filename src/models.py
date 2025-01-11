from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
# from xgboost import XGBClassifier  # If you want to use XGBoost
# from lightgbm import LGBMClassifier # If you want to use LightGBM

def initialize_model(config):
    """Initializes the selected model based on the configuration."""
    model_name = config["train"]["model"]

    if model_name == "IsolationForest":
        return IsolationForest(
            n_estimators=config["train"]["n_estimators"],
            contamination=config["train"]["contamination"],
            random_state=config["train"]["random_state"],
            n_jobs=-1
        )
    elif model_name == "LogisticRegression":
        return LogisticRegression(
            random_state=config["train"]["random_state"],
            n_jobs=-1
        )
    # elif model_name == "XGBClassifier":
    #     return XGBClassifier(
    #         n_estimators=config["train"]["n_estimators"],
    #         random_state=config["train"]["random_state"],
    #         n_jobs=-1
    #     )
    # elif model_name == "LGBMClassifier":
    #     return LGBMClassifier(
    #         n_estimators=config["train"]["n_estimators"],
    #         random_state=config["train"]["random_state"],
    #         n_jobs=-1
    #     )
    else:
        raise ValueError(f"Model {model_name} not recognized.")
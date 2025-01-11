import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

def load_data(data_path, config_path):
    """Loads the credit card fraud dataset and performs initial preprocessing."""
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None, None

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Basic preprocessing
    data['Time'] = data['Time'].apply(lambda x: x / 3600 % 24)  # Convert time to hours of the day
    data = data.drop_duplicates()

    # Split into training and testing sets
    train_data, test_data = train_test_split(
        data, test_size=config["train_test_split"]["test_size"], random_state=config["train_test_split"]["random_state"]
    )

    return train_data, test_data

if __name__ == "__main__":
    train_data, test_data = load_data("data/raw/creditcard.csv", "config.yaml")
    if train_data is not None and test_data is not None:
        train_data.to_csv("data/processed/train_data.csv", index=False)
        test_data.to_csv("data/processed/test_data.csv", index=False)
# Real-time Credit Card Fraud Detection with AI/ML

## Overview

This project implements a real-time credit card fraud detection system using machine learning, specifically focusing on anomaly detection techniques. The system is designed to be production-ready, with a modular code structure, comprehensive testing, and a REST API for real-time predictions.  There is also emphasize explainability, leveraging techniques like SHAP (SHapley Additive exPlanations) to provide insights into why a transaction is flagged as potentially fraudulent.

## Dataset

The project uses the **Credit Card Fraud Detection** dataset from Kaggle: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

*   **Download:** Download the `creditcard.csv` file from Kaggle.
*   **Placement:** Place the downloaded file in the `data/raw/` directory.

The dataset contains transactions made by credit cards in September 2013 by European cardholders. It is highly imbalanced, with only 492 (0.172%) fraud cases out of 284,807 transactions. The features are anonymized (V1-V28) due to privacy reasons and are the result of a PCA transformation. The only non-transformed features are 'Time' (seconds elapsed between each transaction and the first transaction), 'Amount' (transaction amount), and 'Class' (1 for fraud, 0 for genuine).

## Methodology

1.  **Data Ingestion and Preprocessing:**
    *   The `data_ingestion.py` script loads the raw data, performs basic cleaning (e.g., removing duplicates), and splits the data into training and testing sets.
    *   The 'Time' feature is converted to represent the hour of the day.

2.  **Feature Engineering:**
    *   The `feature_engineering.py` script creates new features to enhance the model's ability to detect fraud. These include:
        *   **`Hour`:** Hour of the day derived from the 'Time' feature.
        *   **`Amount_relative_to_mean`:** Ratio of the transaction amount to the overall average transaction amount.
        *   **`Amount_relative_to_mean_hour`:** Ratio of the transaction amount to the average transaction amount for that specific hour.
        *   **`Lagged_Amount`:** (Simplified example) Amount of the previous transaction (based on time).
        *   **`Amount_Diff`:** Difference between the current and previous transaction amounts.
        *   **`Amount_x_Time`:** Interaction feature combining transaction amount and time.
    *   *Note:* More sophisticated features could be engineered if card or user IDs were available.

3.  **Model Training:**
    *   The `training.py` script trains an `IsolationForest` model (by default) for anomaly detection. Other models like `LogisticRegression` are also available in the `models.py` and can be used by changing it in the `config.yaml` file.
    *   The model is trained on the training data and evaluated on a validation set (split from the training data).
    *   Performance metrics like precision, recall, F1-score, and Average Precision are reported.
    *   The trained model and associated files (e.g., `mean_amount`, `mean_amount_by_hour`) are saved in the `models/` directory.

4.  **Real-time Inference and Explainability:**
    *   The `inference.py` script provides functions for:
        *   Loading the trained model and associated data.
        *   Making predictions on new transactions.
        *   Generating SHAP explanations for individual predictions.
    *   The `api.py` script creates a Flask REST API that exposes an endpoint `/predict` for real-time fraud prediction.

5.  **Testing:**
    *   The `tests/` directory contains unit tests for various components of the project, ensuring code correctness and robustness.

## Getting Started

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd fraud-detection-system
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset:**

    *   Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
    *   Place it in the `data/raw/` directory.

5.  **Train the model:**

    ```bash
    python src/training.py
    ```

    This will:
    *   Preprocess the data.
    *   Train the model (using settings from `config.yaml`).
    *   Save the trained model and other necessary files to the `models/` directory.

6.  **Run the API (optional):**

    ```bash
    python src/api.py
    ```

    This will start a Flask development server. You can then send POST requests to `http://0.0.0.0:5000/predict` with transaction data in JSON format to get predictions.

    **Example API Request (using `curl`):**

    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"Time": 100, "V1": -1.3, "V2": 0.8, "V3": 1.5, "V4": 0.4, "V5": -0.5, "V6": 0.9, "V7": -0.8, "V8": 0.5, "V9": -0.3, "V10": 0.2, "V11": -0.6, "V12": 0.7, "V13": 1.2, "V14": -0.9, "V15": 0.1, "V16": -0.4, "V17": 0.6, "V18": -0.2, "V19": 0.3, "V20": 0.1, "V21": -0.2, "V22": 0.5, "V23": -0.1, "V24": 0.0, "V25": 0.2, "V26": -0.3, "V27": 0.1, "V28": 0.05, "Amount": 100}' [http://0.0.0.0:5000/predict](http://0.0.0.0:5000/predict)
    ```

7.  **Run the tests:**

    ```bash
    python -m unittest discover tests
    ```

## Notebooks

The `notebooks/` directory contains Jupyter notebooks that provide more in-depth analysis and experimentation:

*   **`data_exploration.ipynb`:** Exploratory data analysis, visualizations, and insights into the dataset.
*   **`model_development.ipynb`:** Experimentation with different models, hyperparameter tuning, and evaluation.
*   **`explainability.ipynb`:** Demonstrates how to use SHAP to explain model predictions.

## Configuration

The `config.yaml` file contains various parameters that control the behavior of the system:

```yaml
data:
    raw_data_path: "data/raw/creditcard.csv"
    processed_train_data_path: "data/processed/train_data.csv"
    processed_test_data_path: "data/processed/test_data.csv"
train_test_split:
    test_size: 0.2
    random_state: 42
train:
    model: "IsolationForest"  # Options: IsolationForest, LogisticRegression, etc.
    n_estimators: 150        # For Isolation Forest and tree-based models
    contamination: 0.01   # For Isolation Forest - adjust based on your data
    random_state: 42
    val_size: 0.2
model:
    model_path: "models/fraud_detection_model.joblib"
    mean_amount_path: "models/mean_amount.joblib"
    mean_amount_by_hour_path: "models/mean_amount_by_hour.joblib"

from flask import Flask, request, jsonify
from src.inference import load_model, predict
import pandas as pd

app = Flask(__name__)

# Load the model and mean_amount when the app starts
model, mean_amount, mean_amount_by_hour = load_model("config.yaml")

@app.route('/predict', methods=['POST'])
def predict_fraud():
    """API endpoint for making fraud predictions."""
    try:
        transaction_data = request.get_json()
        if transaction_data is None:
            raise ValueError("No transaction data provided.")

        # Convert to DataFrame to handle potential missing values
        transaction_df = pd.DataFrame([transaction_data])

        # Fill missing values with 0 or appropriate defaults
        transaction_df.fillna(0, inplace=True)  # Or use a more sophisticated imputation method

        prediction_result = predict(transaction_df, model, mean_amount, mean_amount_by_hour)
        return jsonify(prediction_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')  # For development, use a production-ready server for deployment
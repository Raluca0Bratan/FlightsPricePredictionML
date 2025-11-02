import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load your trained model
model = joblib.load("../models/random_forest_flight_price_model.pkl")

# List of features expected by the model
FEATURE_COLUMNS = [
    "Airline",
    "AirlineID",
    "Source",
    "Destination",
    "Total_Stops",
    "DurationMinutes",
    "DayOfWeek",
    "IsWeekend",
    "DepartureHour",
    "DepartureDay",
    "DepartureMonth",
    "DepartureWeekday"
]

# Preprocessing function (simple since all features are numeric)
def preprocess_input(data):
    """
    Convert JSON input into a dataframe matching model's feature columns.
    Missing columns will raise an error.
    """
    # Ensure all features are provided
    missing = [col for col in FEATURE_COLUMNS if col not in data]
    if missing:
        raise ValueError(f"Missing features in input: {missing}")

    # Create dataframe in correct order
    df = pd.DataFrame([[data[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    return df


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = preprocess_input(data)
        prediction = model.predict(features)
        return jsonify({
            "input": data,
            "predicted_price": float(prediction[0])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
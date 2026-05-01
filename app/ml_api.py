import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# =========================
# LOAD MODEL + FEATURES
# =========================

model = joblib.load('../models/model.pkl')
features_list = joblib.load('../models/features.pkl')


# =========================
# PREPROCESS FUNCTION
# =========================

def preprocess_input(data, features):
    df = pd.DataFrame([data])

    # verificare coloane lipsă
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    # ordonare exact ca în training
    df = df[features]

    return df.values


# =========================
# PREDICT ENDPOINT
# =========================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # preprocess
        features = preprocess_input(data, features_list)

        # prediction
        prediction = int(model.predict(features)[0])

        # probability (dacă există)
        prob = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            prob = float(probs[prediction])

        return jsonify({
            "decision": "BUY" if prediction == 1 else "WAIT",
            "confidence": prob
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
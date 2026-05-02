import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# =========================
# LOAD MODEL + FEATURES
# =========================

clf = joblib.load('../models/classifier.pkl')
reg = joblib.load('../models/regressor.pkl')

features_list = joblib.load('../models/features.pkl')

route_stats = pd.read_csv('../models/route_stats.csv')
feature_importance = pd.read_csv('../models/feature_importance.csv')


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

        # =====================
        # PREPROCESS
        # =====================
        features = preprocess_input(data, features_list)

        # =====================
        # CLASSIFICATION
        # =====================
        prediction = int(clf.predict(features)[0])
        decision = "BUY" if prediction == 1 else "WAIT"

        prob = None
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(features)[0]
            prob = float(probs[prediction])

        # =====================
        # PRICE PREDICTION (REAL)
        # =====================
        expected_price = float(reg.predict(features)[0])

        # =====================
        # REAL ROUTE STATS
        # =====================
        route = route_stats[
            (route_stats["Source"] == data["Source"]) &
            (route_stats["Destination"] == data["Destination"])
        ]

        if not route.empty:
            stats = {
                "min": float(route["price_min"].values[0]),
                "max": float(route["price_max"].values[0]),
                "avg": float(route["price_avg"].values[0]),
                "median": float(route["price_median"].values[0]),
                "samples": int(route["samples"].values[0])
            }
        else:
            stats = None

        # =====================
        # DEAL QUALITY (REAL)
        # =====================
        deal_quality = None
        price_vs_avg = None

        if stats:
            price_vs_avg = ((expected_price - stats["avg"]) / stats["avg"]) * 100

            deal_quality = (
                "GOOD DEAL" if expected_price < stats["avg"]
                else "EXPENSIVE"
            )

        # =====================
        # RISK LEVEL
        # =====================
        if prob is None:
            risk = "UNKNOWN"
        elif prob > 0.8:
            risk = "LOW"
        elif prob > 0.6:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        # =====================
        # TOP FEATURES (GLOBAL EXPLAINABILITY)
        # =====================
        top_features = feature_importance.head(5).to_dict(orient="records")

        # =====================
        # FINAL RESPONSE
        # =====================
        return jsonify({
            "decision": decision,
            "confidence": prob,
            "risk_level": risk,
            "expected_price": round(expected_price, 2),
            "price_vs_avg_percent": round(price_vs_avg, 2) if price_vs_avg else None,
            "deal_quality": deal_quality,
            "route_stats": stats,
            "top_factors": top_features
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
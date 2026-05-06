import joblib
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# =========================
# LOAD
# =========================
reg = joblib.load('../models/reg_pipeline.pkl')
clf = joblib.load('../models/clf_pipeline.pkl')
route_stats = joblib.load('../models/route_stats.pkl')

# =========================
# PREPROCESS INPUT
# =========================
def preprocess_input(data):
    df = pd.DataFrame([data])

    required = ["airline", "source", "destination", "departuredatetime", "durationminutes", "total_stops"]

    missing = [f for f in required if f not in df.columns]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    # datetime
    df['departuredatetime'] = pd.to_datetime(df['departuredatetime'])

    df['DepartureHour'] = df['departuredatetime'].dt.hour
    df['DepartureDay'] = df['departuredatetime'].dt.day
    df['DepartureMonth'] = df['departuredatetime'].dt.month
    df['DepartureWeekday'] = df['departuredatetime'].dt.weekday
    df["IsWeekend"] = df["DepartureWeekday"].isin([5, 6]).astype(int)

    today = pd.Timestamp.now()
    df["DaysUntilDeparture"] = (df['departuredatetime'] - today).dt.days

    # route stats
    route = route_stats[
        (route_stats["source"] == df["source"].iloc[0]) &
        (route_stats["destination"] == df["destination"].iloc[0])
    ]

    if route.empty:
        raise ValueError("Route not found in stats")

    df["price_avg"] = route["price_avg"].values[0]
    df["price_min"] = route["price_min"].values[0]
    df["price_max"] = route["price_max"].values[0]

    return df

# =========================
# PREDICT
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        X = preprocess_input(data)

        expected_price = float(reg.predict(X)[0])
        ml_decision = int(clf.predict(X)[0])

        prob = float(max(clf.predict_proba(X)[0]))

        price_avg = float(X["price_avg"].values[0])

        # decision
        if ml_decision == 1 and expected_price < price_avg:
            decision = "🔥 STRONG BUY"
        elif ml_decision == 1:
            decision = "BUY"
        elif expected_price > price_avg * 1.1:
            decision = "❌ OVERPRICED"
        else:
            decision = "WAIT"

        return jsonify({
             "decision": decision,
             "confidence": round(prob, 3),
             "expected_price": round(expected_price, 2),
             "price_vs_avg_percent": round(((expected_price - price_avg) / price_avg) * 100, 2),
             "risk_level": "LOW" if prob >= 0.75 else "MEDIUM" if prob >= 0.5 else "HIGH",
             "model_version": "1.0.0",
             "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(port=5001, debug=True)
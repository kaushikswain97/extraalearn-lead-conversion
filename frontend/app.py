"""Flask API for the ExtraaLearn lead-conversion model."""
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask("ExtraaLearn Lead Conversion Predictor")

# Whole pipeline (one-hot encoding + Random Forest) saved in the notebook
model = joblib.load("lead_conversion_model.joblib")

FEATURES = ["age", "current_occupation", "first_interaction", "profile_completed",
            "website_visits", "time_spent_on_website", "page_views_per_visit",
            "last_activity", "print_media_type1", "print_media_type2",
            "digital_media", "educational_channels", "referral"]


@app.get("/")
def home():
    return "Welcome to the ExtraaLearn Lead Conversion Prediction API!"


@app.post("/predict")
def predict_single():
    """Score one lead sent as JSON."""
    payload = request.get_json(silent=True) or {}
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    row = pd.DataFrame([{f: payload[f] for f in FEATURES}])
    prob = float(model.predict_proba(row)[0, 1])
    return jsonify({"conversion_probability": round(prob, 4),
                    "prediction": "Likely to convert" if prob >= 0.5 else "Unlikely to convert"})


@app.post("/predict_batch")
def predict_batch():
    """Score many leads from an uploaded CSV (form field name: file)."""
    if "file" not in request.files:
        return jsonify({"error": "Upload a CSV in the 'file' field"}), 400
    df = pd.read_csv(request.files["file"])
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400
    probs = model.predict_proba(df[FEATURES])[:, 1]
    ids = df["ID"].tolist() if "ID" in df.columns else list(range(len(df)))
    return jsonify({"results": [{"ID": i, "conversion_probability": round(float(p), 4),
                                 "prediction": "Likely to convert" if p >= 0.5 else "Unlikely to convert"}
                                for i, p in zip(ids, probs)]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)

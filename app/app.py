from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/best_random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.route("/")
def home():
    return "House Price Prediction API is running!"


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model": "Random Forest"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "error": "Request body must contain JSON data"
        }), 400
    required_features = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude"
    ]
    missing_features = [
        feature for feature in required_features
        if feature not in data
    ]
    if missing_features:
        return jsonify({
            "error": "Missing required features",
            "missing_features": missing_features
        }), 400

    unexpected_features = [
        feature for feature in data
        if feature not in required_features
    ]

    if unexpected_features:
        return jsonify({
            "error": "Unexpected features provided",
            "unexpected_features": unexpected_features
        }), 400

    invalid_features = [
        feature for feature in required_features
        if not isinstance(data[feature], (int, float))
        or isinstance(data[feature], bool)
    ]

    if invalid_features:
        return jsonify({
            "error": "Feature values must be numbers",
            "invalid_features": invalid_features
        }), 400
    house_data = pd.DataFrame([data])
    house_data_scaled = scaler.transform(house_data)
    prediction = float(model.predict(house_data_scaled)[0])
    predicted_price_dollars = round(prediction * 100000, 2)

    return jsonify({
        "predicted_price": round(prediction, 4),
        "predicted_price_dollars": predicted_price_dollars
    })


if __name__ == "__main__":
    app.run(debug=True)

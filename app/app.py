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

    house_data = pd.DataFrame([data])
    house_data_scaled = scaler.transform(house_data)
    prediction = model.predict(house_data_scaled)
    return jsonify({
        "predicted_price": float(prediction[0])
    })


if __name__ == "__main__":
    app.run(debug=True)

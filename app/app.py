from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


@app.route("/version")
def version():
    return jsonify({
        "api_version": "1.0.0",
        "model_version": "random_forest_v1"
    })


@app.errorhandler(500)
def internal_server_error(error):
    logger.exception("Internal server error")
    return jsonify({
        "error": "Internal server error"
    }), 500


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    logger.info("Prediction request received")

    if data is None:
        return jsonify({
            "error": "Request body must contain JSON data"
        }), 400

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be a JSON object"
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
        logger.warning(
            "Missing required features: %s",
            missing_features
        )
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
    try:
        house_data = pd.DataFrame([data], columns=required_features)
        house_data_scaled = scaler.transform(house_data)
        prediction = float(model.predict(house_data_scaled)[0])
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed"
        }), 500

    logger.info("Prediction completed: %.4f", prediction)
    predicted_price_dollars = round(prediction * 100000, 2)

    return jsonify({
        "predicted_price": round(prediction, 4),
        "predicted_price_dollars": predicted_price_dollars
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

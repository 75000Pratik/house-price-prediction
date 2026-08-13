import time
import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
from flasgger import Swagger

app = Flask(__name__)

swagger = Swagger(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = joblib.load("models/best_random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.route("/")
def home():
    return "House Price Prediction API is running!"


@app.route("/health")
def health():
    """
    Check API health.
    ---
    responses:
      200:
        description: API and model are healthy
        schema:
          type: object
          properties:
            status:
              type: string
              example: healthy
            model:
              type: string
              example: Random Forest
    """
    return jsonify({
        "status": "healthy",
        "model": "Random Forest"
    })


@app.route("/version")
def version():
    """
    Get API and model version information.
    ---
    responses:
      200:
        description: API and model version information
        schema:
          type: object
          properties:
            api_version:
              type: string
              example: "1.0.0"
            model_version:
              type: string
              example: "random_forest_v1"
            build_sha:
              type: string
              example: "43e9d47"  
    """
    return jsonify({
        "api_version": "1.0.0",
        "model_version": "random_forest_v1",
        "build_sha": os.getenv("BUILD_SHA", "local")
    })


@app.errorhandler(500)
def internal_server_error(error):
    logger.exception("Internal server error")
    return jsonify({
        "error": "Internal server error"
    }), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict house price.
    ---
    tags:
      - Prediction
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - MedInc
            - HouseAge
            - AveRooms
            - AveBedrms
            - Population
            - AveOccup
            - Latitude
            - Longitude
          properties:
            MedInc:
              type: number
              example: 8.3252
            HouseAge:
              type: number
              example: 41
            AveRooms:
              type: number
              example: 6.984
            AveBedrms:
              type: number
              example: 1.024
            Population:
              type: number
              example: 322
            AveOccup:
              type: number
              example: 2.556
            Latitude:
              type: number
              example: 37.88
            Longitude:
              type: number
              example: -122.23
    responses:
      200:
        description: House price prediction successful
      400:
        description: Invalid request data
      500:
        description: Internal server error
    """
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
        start_time = time.perf_counter()

        house_data = pd.DataFrame([data], columns=required_features)
        house_data_scaled = scaler.transform(house_data)
        prediction = float(model.predict(house_data_scaled)[0])

        latency = time.perf_counter() - start_time

    except Exception:
        logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed"
        }), 500

    logger.info(
        "Prediction completed: %.4f in %.4f seconds", prediction, latency)

    predicted_price_dollars = round(prediction * 100000, 2)

    return jsonify({
        "predicted_price": round(prediction, 4),
        "predicted_price_dollars": predicted_price_dollars
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/best_random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.route("/")
def home():
    return "House Price Prediction API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    house_data = pd.DataFrame([data])
    house_data_scaled = scaler.transform(house_data)
    prediction = model.predict(house_data_scaled)
    return jsonify({
        "predicted_price": float(prediction[0])
    })


if __name__ == "__main__":
    app.run(debug=True)

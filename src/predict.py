import joblib
import pandas as pd


def load_model():
    model = joblib.load("models/best_random_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


def predict_house_price(model, scaler, data):
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return prediction


if __name__ == "__main__":
    model, scaler = load_model()

    house_data = pd.DataFrame([[
        8.3252,
        41.0,
        6.984127,
        1.023810,
        322.0,
        2.555556,
        37.88,
        -122.23
    ]], columns=[
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude"
    ])
    prediction = predict_house_price(
        model,
        scaler,
        house_data
    )
    print("Predicted House Price:", prediction[0])

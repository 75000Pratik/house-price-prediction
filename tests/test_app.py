from app.app import app


def test_health_endpoint():
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "healthy",
        "model": "Random Forest"
    }


def test_predict_requires_json_body():
    client = app.test_client()

    response = client.post("/predict")

    assert response.status_code == 400
    assert response.get_json(
    )["error"] == "Request body must contain JSON data"


def test_predict_rejects_unexpected_features():
    client = app.test_client()

    response = client.post("/predict", json={
        "MedInc": 8.3252,
        "HouseAge": 41,
        "AveRooms": 6.984,
        "AveBedrms": 1.024,
        "Population": 322,
        "AveOccup": 2.556,
        "Latitude": 37.88,
        "Longitude": -122.23,
        "ExtraFeature": 123
    })

    assert response.status_code == 400
    assert response.get_json()["error"] == "Unexpected features provided"
    assert response.get_json()["unexpected_features"] == ["ExtraFeature"]

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


def test_predict_rejects_non_numeric_features():
    client = app.test_client()

    response = client.post("/predict", json={
        "MedInc": "high",
        "HouseAge": 41,
        "AveRooms": 6.984,
        "AveBedrms": 1.024,
        "Population": 322,
        "AveOccup": 2.556,
        "Latitude": 37.88,
        "Longitude": -122.23
    })

    assert response.status_code == 400
    assert response.get_json()["error"] == "Feature values must be numbers"
    assert response.get_json()["invalid_features"] == ["MedInc"]


def test_predict_returns_price_for_valid_data():
    client = app.test_client()

    response = client.post("/predict", json={
        "MedInc": 8.3252,
        "HouseAge": 41,
        "AveRooms": 6.984,
        "AveBedrms": 1.024,
        "Population": 322,
        "AveOccup": 2.556,
        "Latitude": 37.88,
        "Longitude": -122.23
    })

    response_data = response.get_json()

    assert response.status_code == 200
    assert isinstance(response_data["predicted_price"], float)
    assert isinstance(response_data["predicted_price_dollars"], float)

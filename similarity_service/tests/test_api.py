from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_healthcheck():
    response = client.get("/")
    assert response.status_code == 200
    assert "Servicio de similitud" in response.json()["message"]

def test_prediction_basic():
    payload = {
        "sentence1": "The cat is on the mat",
        "sentence2": "A feline is lying on a rug"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert "similarity_score" in result
    assert "rounded_score" in result
    assert 0 <= result["similarity_score"] <= 5

def test_prediction_invalid():
    # Missing sentence2
    payload = {"sentence1": "Only one sentence"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

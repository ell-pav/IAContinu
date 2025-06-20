import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_generate():
    response = client.post("/generate")
    assert response.status_code == 200
    assert "batch_id" in response.json()

def test_retrain_model():
    client.post("/generate")  # genera un dataset
    response = client.post("/retrain")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict():
    client.post("/generate")
    client.post("/retrain")
    response = client.post("/predict", json={"x1": 0.5, "x2": 0.5})
    assert response.status_code == 200
    assert response.json()["prediction"] in [0, 1]
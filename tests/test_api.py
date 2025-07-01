import httpx
import pytest
from src.api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_predict_endpoint():
    test_data = {
        "recency": 30.0,
        "frequency": 5.0,
        "monetary": 1000.0
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert 0 <= response.json()["risk_probability"] <= 1

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "Production" in response.json()["model_version"]
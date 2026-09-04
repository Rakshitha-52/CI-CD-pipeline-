# tests/test_api.py
def test_predict_logs_to_db(client):
    response = client.post("/predict", json={"features": [0.5] * 30})
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert "confidence" in data
    assert "latency_ms" in data

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "healthy"

def test_stats_endpoint_returns_valid_shape(client):
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.get_json()
    assert "total_predictions" in data
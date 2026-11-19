def test_root_info(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data
    assert "endpoints" in data


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert data["database"] == "connected"
    # The API test suite deliberately stubs inference rather than downloading ML weights.
    assert data["models_loaded"] is False
    assert data["vision_model_loaded"] is False
    assert data["text_model_loaded"] is False
    assert data["available_inference_modes"] == []
    assert "uptime_seconds" in data

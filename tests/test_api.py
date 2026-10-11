import pytest
from fastapi.testclient import TestClient

import nigerian_fake_news.api as api

client = TestClient(api.app)


@pytest.fixture(autouse=True)
def force_untrained_state(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_DIR", tmp_path / "no-model")
    api._load_runtime.cache_clear()
    yield
    api._load_runtime.cache_clear()


def test_health_endpoint_reports_no_untrained_model_as_unavailable():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["model_ready"] is False


def test_predict_requires_training_artifacts():
    response = client.post("/api/predict", json={"text": "This is a sample post for review"})
    assert response.status_code == 503


def test_predict_rejects_too_short_text():
    response = client.post("/api/predict", json={"text": "bad"})
    assert response.status_code == 422

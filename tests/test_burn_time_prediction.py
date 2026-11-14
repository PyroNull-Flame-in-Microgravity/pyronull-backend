import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import burn_time_prediction as prediction_service


client = TestClient(app)


@pytest.fixture
def prediction_payload():
    return {
        "flame_type": "Normal",
        "fuel": "30% ethane",
        "pressure_bar": 0.506650,
        "adiabatic_flame_temperature_k": 2483.47,
        "hydrocarbon_burn_rate_mg_s": 0.75,
        "oxygen_flow_rate_mg_s": 0.85,
        "oxygen_mole_fraction": 0.15,
        "fuel_fraction": 0.300,
    }


def test_actual_model_predicts_documented_sample(prediction_payload):
    prediction_service.load_model.cache_clear()
    try:
        response = client.post("/api/v1/predictions/burn-time", json=prediction_payload)
    finally:
        prediction_service.load_model.cache_clear()

    assert response.status_code == 200
    result = response.json()
    assert result["predicted_burn_time_seconds"] == pytest.approx(196.84, abs=0.01)
    assert result["model_name"] == "PSI-159 Burn Time Regressor"
    assert "Random Forest" in result["model_type"]
    assert result["evaluation_metrics"] == {
        "mae_seconds": 23.08,
        "rmse_seconds": 36.14,
        "r_squared": 0.1575,
    }
    warning = result["experimental_domain_warning"].lower()
    assert "not a fire-risk prediction" in warning
    assert "not validated for lunar, martian, spacecraft" in warning
    assert result["model_limitations"]


def test_model_feature_order_and_field_mapping(prediction_payload, monkeypatch):
    captured = {}

    class MockModel:
        def predict(self, frame):
            captured["columns"] = list(frame.columns)
            captured["values"] = frame.iloc[0].tolist()
            return np.array([12.5])

    monkeypatch.setattr(prediction_service, "load_model", lambda: MockModel())
    response = client.post("/api/v1/predictions/burn-time", json=prediction_payload)
    assert response.status_code == 200
    assert captured["columns"] == [
        "Flame Type",
        "Fuel",
        "approx p (bar)",
        "Tad (K)",
        "HC burnrate (mg/s)",
        "O2 flow rate (mg/s)",
        "XO2",
        "Xf",
    ]
    assert captured["values"] == [
        "Normal", "30% ethane", 0.506650, 2483.47, 0.75, 0.85, 0.15, 0.300,
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("pressure_bar", float("nan")),
        ("adiabatic_flame_temperature_k", float("inf")),
        ("hydrocarbon_burn_rate_mg_s", float("-inf")),
        ("oxygen_flow_rate_mg_s", float("nan")),
        ("oxygen_mole_fraction", float("inf")),
        ("fuel_fraction", float("nan")),
    ],
)
def test_non_finite_numerics_are_rejected(prediction_payload, field, value):
    prediction_payload[field] = value
    response = client.post(
        "/api/v1/predictions/burn-time",
        content=json.dumps(prediction_payload, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("pressure_bar", 0),
        ("adiabatic_flame_temperature_k", -1),
        ("hydrocarbon_burn_rate_mg_s", -0.01),
        ("oxygen_mole_fraction", 1.01),
        ("fuel_fraction", -0.01),
        ("flame_type", "   "),
        ("fuel", ""),
    ],
)
def test_out_of_domain_or_blank_inputs_are_rejected(prediction_payload, field, value):
    prediction_payload[field] = value
    response = client.post("/api/v1/predictions/burn-time", json=prediction_payload)
    assert response.status_code == 422


def test_model_is_loaded_once_and_reused(tmp_path, monkeypatch):
    calls = []
    fake_model = SimpleNamespace(feature_names_in_=prediction_service.MODEL_FEATURES)
    monkeypatch.setattr(prediction_service, "MODEL_PATH", tmp_path / "model.joblib")
    (tmp_path / "model.joblib").write_bytes(b"mock artifact")
    monkeypatch.setattr(prediction_service.joblib, "load", lambda path: calls.append(path) or fake_model)
    prediction_service.load_model.cache_clear()
    try:
        assert prediction_service.load_model() is fake_model
        assert prediction_service.load_model() is fake_model
        assert len(calls) == 1
    finally:
        prediction_service.load_model.cache_clear()


def test_missing_model_returns_service_unavailable(prediction_payload, tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_service, "MODEL_PATH", tmp_path / "missing.joblib")
    prediction_service.load_model.cache_clear()
    try:
        response = client.post("/api/v1/predictions/burn-time", json=prediction_payload)
    finally:
        prediction_service.load_model.cache_clear()
    assert response.status_code == 503
    assert "artifact is not installed" in response.json()["detail"]


def test_model_loading_error_returns_service_unavailable(prediction_payload, tmp_path, monkeypatch):
    model_path = tmp_path / "broken.joblib"
    model_path.write_bytes(b"broken")
    monkeypatch.setattr(prediction_service, "MODEL_PATH", model_path)
    monkeypatch.setattr(prediction_service.joblib, "load", lambda path: (_ for _ in ()).throw(ValueError("bad artifact")))
    prediction_service.load_model.cache_clear()
    try:
        response = client.post("/api/v1/predictions/burn-time", json=prediction_payload)
    finally:
        prediction_service.load_model.cache_clear()
    assert response.status_code == 503
    assert "could not be loaded" in response.json()["detail"]


@pytest.mark.parametrize("prediction", [[float("inf")], []])
def test_invalid_model_output_returns_controlled_error(prediction_payload, monkeypatch, prediction):
    class MockModel:
        def predict(self, frame):
            return np.asarray(prediction)

    monkeypatch.setattr(prediction_service, "load_model", lambda: MockModel())
    response = client.post("/api/v1/predictions/burn-time", json=prediction_payload)
    assert response.status_code == 500


def test_prediction_route_is_documented_with_typed_response_schema():
    openapi = app.openapi()
    operation = openapi["paths"]["/api/v1/predictions/burn-time"]["post"]
    assert "BurnTimePredictionRequest" in operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    assert "BurnTimePredictionResponse" in operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    assert client.get("/docs").status_code == 200


def test_existing_mission_intelligence_and_other_routes_remain_registered():
    paths = app.openapi()["paths"]
    assert "/api/v1/mission-intelligence/profiles" in paths
    assert "/api/v1/investigations" in paths
    assert "/api/v1/compare" in paths
    assert "/api/v1/assistant/query" in paths
    assert "/api/v1/rank" in paths

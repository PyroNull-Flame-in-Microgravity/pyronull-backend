from fastapi.testclient import TestClient
from app.main import _configured_cors_origins, app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_cors_origins_default_to_local_development_frontends():
    assert _configured_cors_origins({}) == [
        "http://localhost:3000",
        "http://localhost:5173",
    ]


def test_cors_origins_can_be_configured_as_comma_separated_origins():
    assert _configured_cors_origins({
        "CORS_ALLOWED_ORIGINS": " https://app.example.org/ , https://preview.example.org "
    }) == ["https://app.example.org", "https://preview.example.org"]


def test_cors_middleware_accepts_development_origin():
    response = client.options(
        "/api/v1/predictions/burn-time",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

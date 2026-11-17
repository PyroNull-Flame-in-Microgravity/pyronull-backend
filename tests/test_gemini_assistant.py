import sys
from types import SimpleNamespace
from types import ModuleType

from fastapi.testclient import TestClient
import pytest

from app.core.config import DEFAULT_GEMINI_MODEL
from app.main import app
from app.services import gemini_assistant


client = TestClient(app)


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response or SimpleNamespace(text="Only the supplied fixture supports this summary.")
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, models):
        self.models = models


@pytest.fixture(autouse=True)
def clean_gemini_state(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    monkeypatch.setenv("GEMINI_TIMEOUT_MS", "25000")
    monkeypatch.setenv("GEMINI_REQUESTS_PER_MINUTE", "30")
    gemini_assistant.clear_client_cache()
    gemini_assistant.clear_rate_limit_state()
    yield
    gemini_assistant.clear_client_cache()
    gemini_assistant.clear_rate_limit_state()


def install_fake_client(monkeypatch, models):
    seen = {}

    def build(api_key, timeout_ms):
        seen["api_key"] = api_key
        seen["timeout_ms"] = timeout_ms
        return FakeClient(models)

    monkeypatch.setenv("GEMINI_API_KEY", "test-only-secret-value")
    monkeypatch.setattr(gemini_assistant, "_build_gemini_client", build)
    monkeypatch.setattr(gemini_assistant, "_generation_config", lambda: {"temperature": 0.2})
    return seen


def test_client_initialization_uses_env_key_timeout_and_single_sdk_attempt(monkeypatch):
    fake_google = ModuleType("google")
    fake_genai = ModuleType("google.genai")
    fake_types = ModuleType("google.genai.types")
    fake_types.HttpOptions = lambda **kwargs: kwargs
    fake_types.HttpRetryOptions = lambda **kwargs: kwargs
    fake_genai.types = fake_types
    fake_genai.Client = lambda **kwargs: SimpleNamespace(**kwargs)
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    client_object = gemini_assistant._build_gemini_client("test-only-secret-value", 15000)
    assert client_object.api_key == "test-only-secret-value"
    assert client_object.http_options == {
        "timeout": 15000,
        "retry_options": {"attempts": 1},
    }


def test_missing_api_key_does_not_break_backend_and_ai_route_returns_configuration_error():
    assert client.get("/health").status_code == 200
    response = client.post(
        "/api/v1/assistant/query",
        json={"question": "Summarize the combustion investigation", "investigation_ids": ["PSI-69"]},
    )
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]
    assert "test-only-secret-value" not in response.text


def test_assistant_returns_grounded_sources_and_does_not_expose_key(monkeypatch):
    models = FakeModels()
    seen = install_fake_client(monkeypatch, models)
    response = client.post(
        "/api/v1/assistant/query",
        json={"question": "Summarize this combustion investigation", "investigation_ids": ["PSI-69"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "sources", "limitations", "status"}
    assert body["status"] == "GENERATED"
    assert body["sources"][0]["source_id"] == "PSI-69"
    assert body["sources"][0]["verification_status"] == "UNVERIFIED"
    assert seen == {"api_key": "test-only-secret-value", "timeout_ms": 25000}
    assert "Do not browse" in models.calls[0]["contents"]
    assert models.calls[0]["model"] == DEFAULT_GEMINI_MODEL
    assert "PSI-69" in models.calls[0]["contents"]
    assert "test-only-secret-value" not in response.text


def test_non_allowlisted_paid_model_fails_closed_without_fallback(monkeypatch):
    models = FakeModels()
    install_fake_client(monkeypatch, models)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-4-argon")

    response = client.post(
        "/api/v1/assistant/query",
        json={"question": "Summarize this combustion investigation", "investigation_ids": ["PSI-69"]},
    )

    assert response.status_code == 503
    assert models.calls == []
    assert "configuration is invalid" in response.json()["detail"]
    assert "gemini-4-argon" not in response.text


def test_no_relevant_context_returns_without_calling_gemini():
    response = client.post("/api/v1/assistant/query", json={"question": "zzzz qqqqq xxxxxx"})
    assert response.status_code == 200
    assert response.json()["status"] == "NO_CONTEXT"
    assert response.json()["answer"] is None
    assert response.json()["sources"] == []


def test_summary_endpoint_uses_existing_fixture_record(monkeypatch):
    models = FakeModels(SimpleNamespace(text="Objective: not reported. Findings: not reported."))
    install_fake_client(monkeypatch, models)
    response = client.post("/api/v1/ai/research/summarize", json={"investigation_id": "PSI-69"})
    assert response.status_code == 200
    body = response.json()
    assert body["sources"][0]["source_id"] == "PSI-69"
    assert "demo CSV fixtures" in " ".join(body["limitations"])
    assert "Not reported" in models.calls[0]["contents"]


def test_unknown_summary_record_returns_404(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-secret-value")
    response = client.post("/api/v1/ai/research/summarize", json={"investigation_id": "UNKNOWN-ID"})
    assert response.status_code == 404


def test_mission_explain_preserves_existing_numeric_results(monkeypatch):
    models = FakeModels(SimpleNamespace(text="This is a fixture-scoped relevance explanation."))
    install_fake_client(monkeypatch, models)
    response = client.post(
        "/api/v1/ai/mission/explain",
        json={"scenario": {"mission_environment": "lunar_habitat"}, "research_question": "flame combustion", "limit": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ranked_investigations"]
    assert body["ranked_investigations"][0]["score_label"] == "HEURISTIC_RESEARCH_RELEVANCE_NOT_FIRE_RISK_OR_SAFETY_SCORE"
    assert body["data_status"] == "DEMO_DATA_NOT_VERIFIED_NASA_RESEARCH"
    assert "Preserve the values" in models.calls[0]["contents"]


def test_mission_report_contains_structured_report_and_narrative(monkeypatch):
    models = FakeModels(SimpleNamespace(text="Scenario; evidence; gaps; limitations."))
    install_fake_client(monkeypatch, models)
    response = client.post(
        "/api/v1/ai/mission/report",
        json={"scenario": {"mission_environment": "space_station"}, "research_question": "flame spread", "limit": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["report"]
    assert body["structured_report"]["research_gaps"]["absence_of_research_claimed"] is False
    assert "never operational safety" in models.calls[0]["contents"].lower()


def test_gap_explanation_includes_original_fixture_scoped_gaps(monkeypatch):
    install_fake_client(monkeypatch, FakeModels(SimpleNamespace(text="This only describes the provided fixtures.")))
    response = client.post(
        "/api/v1/ai/research/gaps/explain",
        json={"mission_environment": "mars_habitat", "gravity_condition": "partial gravity"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["research_gaps"]["scope"] == "provided_csv_fixtures_only"
    assert body["research_gaps"]["absence_of_research_claimed"] is False
    assert body["sources"] == []


@pytest.mark.parametrize(
    "error,expected_status",
    [
        (
            type(
                "RateLimitError",
                (Exception,),
                {"status_code": 429, "response": SimpleNamespace(headers={"Retry-After": "17"})},
            )(),
            429,
        ),
        (type("APITimeoutError", (Exception,), {})(), 504),
        (type("ServerError", (Exception,), {"status_code": 500})(), 502),
    ],
)
def test_provider_failures_map_to_safe_http_statuses(monkeypatch, error, expected_status):
    install_fake_client(monkeypatch, FakeModels(error=error))
    response = client.post("/api/v1/ai/research/summarize", json={"investigation_id": "PSI-69"})
    assert response.status_code == expected_status
    assert "test-only-secret-value" not in response.text
    if expected_status == 429:
        assert response.headers["retry-after"] == "17"


def test_in_memory_rate_limit_is_enforced(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-secret-value")
    monkeypatch.setenv("GEMINI_REQUESTS_PER_MINUTE", "1")
    gemini_assistant._acquire_rate_slot()
    with pytest.raises(gemini_assistant.GeminiRateLimitError):
        gemini_assistant._acquire_rate_slot()


def test_new_ai_routes_have_openapi_schemas_and_invalid_inputs_are_rejected():
    paths = app.openapi()["paths"]
    assert "/api/v1/ai/research/summarize" in paths
    assert "/api/v1/ai/mission/explain" in paths
    assert "/api/v1/ai/mission/report" in paths
    assert "/api/v1/ai/research/gaps/explain" in paths
    assert client.post("/api/v1/ai/research/summarize", json={}).status_code == 422
    assert client.post("/api/v1/assistant/query", json={"question": "x"}).status_code == 422


def test_existing_prediction_paths_are_still_registered():
    paths = app.openapi()["paths"]
    assert "/api/v1/predictions/burn-time" in paths
    assert "/api/v1/predictions/psi10/adiabatic-flame-temperature" in paths
    assert "/api/v1/predictions/psi10/stoichiometric-mixture-fraction" in paths
    assert "/api/v1/predictions/psi10/pressure-rise" in paths
    assert "/api/v1/predictions/psi159/ignition" in paths
    assert "/api/v1/predictions/psi107/fuel-classification" in paths
    assert "/api/v1/predictions/psi69/flame-outcome" in paths

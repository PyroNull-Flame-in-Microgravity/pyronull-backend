import json

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.models.mission_intelligence import MissionEnvironment, MissionQueryRequest, MissionScenarioRequest
from app.services import mission_intelligence as mission_service
from app.services.mission_intelligence import DatasetLoadError, identify_gaps, load_investigations, rank_investigations


client = TestClient(app)


def _payload():
    return {
        "scenario": {
            "mission_environment": "lunar_habitat",
            "oxygen_concentration_pct": 21,
            "pressure_kpa": 101.3,
            "gravity_condition": "partial gravity",
            "airflow_velocity_m_s": 0.2,
            "habitat_type": "pressurized habitat",
            "fuel_material": "PMMA",
        },
        "research_question": "flame spread fire safety",
        "safety_domain": "fire safety",
        "limit": 5,
    }


def test_profiles_endpoint_lists_five_supported_environments():
    response = client.get("/api/v1/mission-intelligence/profiles")
    assert response.status_code == 200
    profiles = response.json()
    assert len(profiles) == 5
    assert {profile["id"] for profile in profiles} == {item.value for item in MissionEnvironment}
    assert all("dataset_applicability_note" in profile for profile in profiles)


def test_scenario_endpoint_keeps_omitted_conditions_unknown():
    response = client.post(
        "/api/v1/mission-intelligence/scenario",
        json={"mission_environment": "mars_habitat"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["known_conditions"] == []
    assert "oxygen_concentration_pct" in result["explicitly_unknown_conditions"]
    assert result["scenario"]["pressure_kpa"] is None


def test_scenario_rejects_out_of_range_oxygen():
    response = client.post(
        "/api/v1/mission-intelligence/scenario",
        json={"mission_environment": "spacecraft", "oxygen_concentration_pct": 120},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("oxygen_concentration_pct", float("nan")),
        ("oxygen_concentration_pct", float("inf")),
        ("oxygen_concentration_pct", float("-inf")),
        ("pressure_kpa", float("nan")),
        ("pressure_kpa", float("inf")),
        ("pressure_kpa", float("-inf")),
        ("airflow_velocity_m_s", float("nan")),
        ("airflow_velocity_m_s", float("inf")),
        ("airflow_velocity_m_s", float("-inf")),
    ],
)
def test_scenario_rejects_non_finite_numbers_with_422(field, value):
    payload = {"mission_environment": "spacecraft", field: value}
    response = client.post(
        "/api/v1/mission-intelligence/scenario",
        content=json.dumps(payload, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_discover_returns_dataset_records_and_provenance():
    response = client.post("/api/v1/mission-intelligence/discover", json=_payload())
    assert response.status_code == 200
    result = response.json()
    assert result["total"] == 11
    assert result["returned"] == 5
    assert result["data_status"] == "DEMO_DATA_NOT_VERIFIED_NASA_RESEARCH"
    assert {item["psi_record_id"] for item in result["items"]}.issubset({row["psi_record_id"] for row in load_investigations()})
    assert all(item["provenance"]["verification_status"] == "not independently verified" for item in result["items"])


def test_rank_endpoint_exposes_explanations_and_non_safety_score_label():
    response = client.post("/api/v1/mission-intelligence/rank", json=_payload())
    assert response.status_code == 200
    ranked = response.json()["ranked_investigations"]
    assert ranked
    assert ranked[0]["score_label"] == "HEURISTIC_RESEARCH_RELEVANCE_NOT_FIRE_RISK_OR_SAFETY_SCORE"
    assert set(ranked[0]["explanations"]) == {
        "condition_similarity",
        "material_fuel_relevance",
        "safety_domain_relevance",
        "measurement_availability",
        "condition_applicability_coverage",
        "mission_environment_applicability",
    }
    assert ranked[0]["dimension_scores"]["mission_environment_applicability"] is None
    assert ranked[0]["dimension_scores"]["condition_similarity"] is not None
    assert ranked[0]["dimension_scores"]["measurement_availability"] is None


def test_reported_oxygen_ranges_are_parsed_as_ranges_not_negative_values():
    request = MissionQueryRequest(
        scenario=MissionScenarioRequest(mission_environment="spacecraft", oxygen_concentration_pct=21),
        limit=50,
    )
    by_id = {item["psi_record_id"]: item for item in rank_investigations(request)}
    flex_oxygen = next(
        item for item in by_id["PSI-69"]["explanations"]["condition_similarity"]
        if item["condition"] == "oxygen_concentration_pct"
    )
    assert flex_oxygen["dataset_value"] == "10-30%"
    assert flex_oxygen["status"] == "within_reported_range"
    assert flex_oxygen["similarity"] == 1.0


def test_unit_parser_requires_explicit_units_and_converts_airflow():
    assert mission_service._numeric("21", "oxygen") is None
    assert mission_service._numeric("101.3", "pressure") is None
    assert mission_service._numeric("5 cm/s", "airflow") == (0.05, 0.05)
    assert mission_service._numeric("5 mm/s", "airflow") == (0.005, 0.005)
    assert mission_service._numeric("5 cm/s", "airflow") != (5.0, 5.0)


def test_core_ranking_does_not_invent_unknown_measurements():
    request = MissionQueryRequest(scenario=MissionScenarioRequest(mission_environment="spacecraft"), limit=50)
    ranked = rank_investigations(request)
    assert len(ranked) == 11
    assert all(not item["measurement_available_in_fixture"] for item in ranked)
    assert all(item["dimension_scores"]["condition_similarity"] is None for item in ranked)
    assert all(item["dimension_scores"]["measurement_availability"] is None for item in ranked)
    assert all(item["measurement_status_in_fixture"] == "not_reported_unknown" for item in ranked)
    assert all(item["dimension_scores"]["mission_environment_applicability"] is None for item in ranked)


def test_rank_score_is_unknown_when_no_dimensions_can_be_evaluated():
    request = MissionQueryRequest(scenario=MissionScenarioRequest(mission_environment="spacecraft"))
    ranked = rank_investigations(request)
    assert all(item["relevance_score"] is None for item in ranked)


def test_gaps_distinguish_fixture_coverage_from_absence_of_research():
    scenario = MissionScenarioRequest(
        mission_environment="mars_habitat",
        gravity_condition="partial gravity",
        airflow_velocity_m_s=0.2,
    )
    result = identify_gaps(scenario)
    assert result["scope"] == "provided_csv_fixtures_only"
    assert result["absence_of_research_claimed"] is False
    assert any(gap["dimension"] == "mission_environment_applicability" for gap in result["gaps"])
    assert all("does not demonstrate absence" in gap["interpretation"] for gap in result["gaps"])


def test_gaps_endpoint_returns_fixture_scoped_gaps():
    response = client.post(
        "/api/v1/mission-intelligence/gaps",
        json={"mission_environment": "pressurized_rover", "gravity_condition": "partial gravity"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["scope"] == "provided_csv_fixtures_only"
    assert result["absence_of_research_claimed"] is False


def test_report_contains_profile_evidence_limitations_and_gaps():
    response = client.post("/api/v1/mission-intelligence/report", json=_payload())
    assert response.status_code == 200
    result = response.json()
    assert result["profile"]["id"] == "lunar_habitat"
    assert result["ranked_investigations"]
    assert result["evidence"]
    assert result["limitations"]
    assert result["research_gaps"]["absence_of_research_claimed"] is False


def test_all_mission_endpoints_are_in_openapi_and_existing_routes_remain():
    openapi = app.openapi()
    paths = openapi["paths"]
    expected = {
        "/api/v1/mission-intelligence/profiles",
        "/api/v1/mission-intelligence/scenario",
        "/api/v1/mission-intelligence/discover",
        "/api/v1/mission-intelligence/rank",
        "/api/v1/mission-intelligence/gaps",
        "/api/v1/mission-intelligence/report",
    }
    assert expected.issubset(paths)
    assert "/api/v1/investigations" in paths
    assert "/api/v1/compare" in paths
    assert "/api/v1/assistant/query" in paths
    assert "/api/v1/rank" in paths
    for path in expected:
        method = "get" if path.endswith("/profiles") else "post"
        schema = paths[path][method]["responses"]["200"]["content"]["application/json"]["schema"]
        if path.endswith("/profiles"):
            schema = schema["items"]
        assert "$ref" in schema


def test_existing_routes_still_respond_and_assistant_keeps_its_response_shape(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert client.get("/api/v1/investigations").status_code == 200
    compare_response = client.post("/api/v1/compare", json={"investigation_ids": ["demo-a", "demo-b"]})
    assert compare_response.status_code == 200
    compare_body = compare_response.json()
    assert compare_body == {
        "requested_ids": ["demo-a", "demo-b"],
        "common_fields": [],
        "differences": [],
        "warnings": [
            "Not connected to verified NASA source data.",
            "Do not treat unlike experiments as controlled head-to-head trials.",
        ],
        "status": "NOT_IMPLEMENTED",
    }
    assistant_response = client.post(
        "/api/v1/assistant/query",
        json={"question": "Find combustion evidence", "investigation_ids": ["PSI-69"]},
    )
    assert assistant_response.status_code == 503
    assert "GEMINI_API_KEY" in assistant_response.json()["detail"]
    rank_response = client.post("/api/v1/rank", json={"research_question": "flame spread"})
    assert rank_response.status_code == 200
    assert rank_response.json() == {
        "research_question": "flame spread",
        "ranked_items": [],
        "scoring_method": {
            "objective_match": 0.40,
            "condition_similarity": 0.25,
            "evidence_completeness": 0.20,
            "mission_context_fit": 0.15,
        },
        "score_label": "RESEARCH_RELEVANCE_NOT_FIRE_RISK",
        "status": "NOT_IMPLEMENTED",
    }


def test_loader_tolerates_missing_optional_columns_and_missing_cells(tmp_path, monkeypatch):
    full = tmp_path / "full.csv"
    investigations = tmp_path / "investigations.csv"
    full.write_text("psi_record_id,fuel_material,measurement_method\nPSI-TEST,PMMA,imaging\n", encoding="utf-8")
    investigations.write_text("psi_record_id,investigation_title,fuel_material\nPSI-TEST,Example,\n", encoding="utf-8")
    monkeypatch.setattr(mission_service, "FULL_DATASET_CSV", full)
    monkeypatch.setattr(mission_service, "INVESTIGATIONS_CSV", investigations)
    load_investigations.cache_clear()
    try:
        records = load_investigations()
    finally:
        load_investigations.cache_clear()
    assert len(records) == 1
    assert records[0]["psi_record_id"] == "PSI-TEST"
    assert records[0]["fuel_material"] == "PMMA"
    assert records[0]["measurement_method"] == "imaging"
    assert records[0].get("airflow_velocity") is None


@pytest.mark.parametrize(
    "content",
    [
        "fuel_material\nPMMA\n",
        "psi_record_id,fuel_material\nPSI-TEST,PMMA,EXTRA\n",
        "psi_record_id,psi_record_id\nPSI-TEST,PSI-OTHER\n",
        'psi_record_id,fuel_material\nPSI-TEST,"unterminated\n',
    ],
)
def test_loader_rejects_malformed_dataset_structure(tmp_path, monkeypatch, content):
    full = tmp_path / "full.csv"
    investigations = tmp_path / "investigations.csv"
    full.write_text(content, encoding="utf-8")
    investigations.write_text("psi_record_id\nPSI-TEST\n", encoding="utf-8")
    monkeypatch.setattr(mission_service, "FULL_DATASET_CSV", full)
    monkeypatch.setattr(mission_service, "INVESTIGATIONS_CSV", investigations)
    load_investigations.cache_clear()
    try:
        with pytest.raises(DatasetLoadError):
            load_investigations()
    finally:
        load_investigations.cache_clear()


def test_missing_dataset_is_reported_as_service_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(mission_service, "FULL_DATASET_CSV", tmp_path / "missing.csv")
    load_investigations.cache_clear()
    try:
        response = client.post("/api/v1/mission-intelligence/discover", json=_payload())
    finally:
        load_investigations.cache_clear()
    assert response.status_code == 503
    assert "unavailable or invalid" in response.json()["detail"]

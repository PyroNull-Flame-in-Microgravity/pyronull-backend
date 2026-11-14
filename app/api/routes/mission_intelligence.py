from fastapi import APIRouter, HTTPException

from app.models.mission_intelligence import (
    MissionProfile,
    MissionDiscoveryResponse,
    MissionGapsResponse,
    MissionQueryRequest,
    MissionRankResponse,
    MissionReportResponse,
    MissionScenarioRequest,
    MissionScenarioResponse,
)
from app.services.mission_intelligence import (
    DATA_STATUS,
    DatasetLoadError,
    LIMITATIONS,
    build_report,
    discover_investigations,
    get_profiles,
    identify_gaps,
    rank_investigations,
    scenario_summary,
)


router = APIRouter()


def _run_dataset_operation(operation, *args):
    try:
        return operation(*args)
    except DatasetLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/mission-intelligence/profiles", response_model=list[MissionProfile])
def mission_profiles():
    return get_profiles()


@router.post("/mission-intelligence/scenario", response_model=MissionScenarioResponse)
def configure_scenario(payload: MissionScenarioRequest):
    return scenario_summary(payload)


@router.post("/mission-intelligence/discover", response_model=MissionDiscoveryResponse)
def discover(payload: MissionQueryRequest):
    return _run_dataset_operation(discover_investigations, payload)


@router.post("/mission-intelligence/rank", response_model=MissionRankResponse)
def rank(payload: MissionQueryRequest):
    return {
        "scenario": payload.scenario.model_dump(mode="json"),
        "ranked_investigations": _run_dataset_operation(rank_investigations, payload),
        "scoring_notice": "Heuristic research relevance only; not a validated fire-risk or safety score.",
        "data_status": DATA_STATUS,
        "limitations": LIMITATIONS,
    }


@router.post("/mission-intelligence/gaps", response_model=MissionGapsResponse)
def gaps(payload: MissionScenarioRequest):
    return _run_dataset_operation(identify_gaps, payload)


@router.post("/mission-intelligence/report", response_model=MissionReportResponse)
def report(payload: MissionQueryRequest):
    return _run_dataset_operation(build_report, payload)
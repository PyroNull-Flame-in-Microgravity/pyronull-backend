"""Gemini-powered summaries and explanations over existing mission services."""

from fastapi import APIRouter, HTTPException

from app.models.mission_intelligence import MissionQueryRequest, MissionScenarioRequest
from app.schemas.gemini import (
    GeneratedTextResponse,
    MissionExplanationResponse,
    MissionReportAIResponse,
    ResearchGapsAIResponse,
    ResearchSummaryRequest,
)
from app.services.gemini_assistant import (
    DATA_LIMITATIONS,
    GeminiConfigurationError,
    GeminiProviderError,
    GeminiRateLimitError,
    GeminiTimeoutError,
    ResearchRecordNotFound,
    explain_mission as create_mission_explanation,
    explain_research_gaps,
    generate_mission_report,
    summarize_investigation,
)
from app.services.mission_intelligence import DATA_STATUS, DatasetLoadError


router = APIRouter()


def _raise_ai_http_error(exc):
    if isinstance(exc, GeminiConfigurationError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, GeminiRateLimitError):
        raise HTTPException(
            status_code=429,
            detail="AI request limit reached. Try again later.",
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    if isinstance(exc, GeminiTimeoutError):
        raise HTTPException(status_code=504, detail="Gemini request timed out.") from exc
    if isinstance(exc, GeminiProviderError):
        raise HTTPException(status_code=502, detail="Gemini could not complete the request.") from exc
    if isinstance(exc, ResearchRecordNotFound):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, DatasetLoadError):
        raise HTTPException(status_code=503, detail="Research fixture data is unavailable.") from exc
    raise exc


def _run_ai(operation):
    try:
        return operation()
    except (
        GeminiConfigurationError,
        GeminiRateLimitError,
        GeminiTimeoutError,
        GeminiProviderError,
        ResearchRecordNotFound,
        DatasetLoadError,
    ) as exc:
        _raise_ai_http_error(exc)


@router.post("/ai/research/summarize", response_model=GeneratedTextResponse)
def summarize_research(payload: ResearchSummaryRequest):
    content, sources = _run_ai(lambda: summarize_investigation(payload.investigation_id))
    return GeneratedTextResponse(content=content, sources=sources, limitations=DATA_LIMITATIONS, status="GENERATED")


@router.post("/ai/mission/explain", response_model=MissionExplanationResponse)
def explain_mission(payload: MissionQueryRequest):
    explanation, ranked, sources = _run_ai(lambda: create_mission_explanation(payload))
    return MissionExplanationResponse(
        explanation=explanation,
        ranked_investigations=ranked,
        sources=sources,
        data_status=DATA_STATUS,
        limitations=DATA_LIMITATIONS,
    )


@router.post("/ai/mission/report", response_model=MissionReportAIResponse)
def mission_report(payload: MissionQueryRequest):
    report, structured_report, sources = _run_ai(lambda: generate_mission_report(payload))
    return MissionReportAIResponse(
        report=report,
        structured_report=structured_report,
        sources=sources,
        data_status=DATA_STATUS,
        limitations=DATA_LIMITATIONS,
    )


@router.post("/ai/research/gaps/explain", response_model=ResearchGapsAIResponse)
def explain_gaps(payload: MissionScenarioRequest):
    result = _run_ai(lambda: explain_research_gaps(payload))
    return ResearchGapsAIResponse(
        explanation=result["explanation"],
        research_gaps=result["research_gaps"],
        data_status=DATA_STATUS,
        limitations=DATA_LIMITATIONS,
    )

"""Structured request and response contracts for Gemini-backed features."""

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

from app.models.mission_intelligence import MissionGapsResponse, RankedInvestigation


class ResearchSource(BaseModel):
    source_id: str
    title: str
    source_url: Optional[str] = None
    verification_status: str
    data_status: str


class AssistantResponse(BaseModel):
    answer: Optional[str] = None
    sources: List[ResearchSource] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    status: Literal["GENERATED", "NO_CONTEXT"]


class ResearchSummaryRequest(BaseModel):
    investigation_id: str = Field(min_length=2, max_length=80)


class GeneratedTextResponse(BaseModel):
    content: str
    sources: List[ResearchSource] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    status: Literal["GENERATED", "NO_CONTEXT"]


class MissionExplanationResponse(BaseModel):
    explanation: str
    ranked_investigations: List[RankedInvestigation]
    sources: List[ResearchSource] = Field(default_factory=list)
    data_status: str
    limitations: List[str] = Field(default_factory=list)
    status: Literal["GENERATED"] = "GENERATED"


class MissionReportAIResponse(BaseModel):
    report: str
    structured_report: dict[str, Any]
    sources: List[ResearchSource] = Field(default_factory=list)
    data_status: str
    limitations: List[str] = Field(default_factory=list)
    status: Literal["GENERATED"] = "GENERATED"


class ResearchGapsAIResponse(BaseModel):
    explanation: str
    research_gaps: MissionGapsResponse
    sources: List[ResearchSource] = Field(default_factory=list)
    data_status: str
    limitations: List[str] = Field(default_factory=list)
    status: Literal["GENERATED"] = "GENERATED"

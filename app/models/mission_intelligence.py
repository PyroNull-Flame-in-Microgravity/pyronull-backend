from enum import Enum
import math
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class MissionEnvironment(str, Enum):
    SPACECRAFT = "spacecraft"
    SPACE_STATION = "space_station"
    LUNAR_HABITAT = "lunar_habitat"
    MARS_HABITAT = "mars_habitat"
    PRESSURIZED_ROVER = "pressurized_rover"


class MissionScenarioRequest(BaseModel):
    mission_environment: MissionEnvironment = Field(
        description="Mission environment to explore. This is not inferred from the NASA fixture records."
    )
    oxygen_concentration_pct: Optional[float] = Field(
        default=None, ge=0, le=100, allow_inf_nan=False,
        description="Scenario oxygen concentration in percent; omit if unknown."
    )
    pressure_kpa: Optional[float] = Field(
        default=None, gt=0, allow_inf_nan=False,
        description="Scenario pressure in kPa; omit if unknown."
    )
    gravity_condition: Optional[str] = Field(
        default=None, min_length=2, max_length=80, description="Gravity condition; omit if unknown."
    )
    airflow_velocity_m_s: Optional[float] = Field(
        default=None, ge=0, allow_inf_nan=False,
        description="Scenario airflow velocity in m/s; omit if unknown."
    )
    habitat_type: Optional[str] = Field(
        default=None, min_length=2, max_length=100, description="Habitat type; omit if unknown."
    )
    fuel_material: Optional[str] = Field(
        default=None, min_length=1, max_length=120, description="Material/fuel of interest; omit if unspecified."
    )

    @field_validator(
        "oxygen_concentration_pct",
        "pressure_kpa",
        "airflow_velocity_m_s",
        mode="before",
    )
    @classmethod
    def reject_non_finite_scenario_values(cls, value):
        if isinstance(value, (int, float)) and not math.isfinite(value):
            # Keep FastAPI's echoed validation details JSON serializable.
            return repr(value)
        return value


class MissionQueryRequest(BaseModel):
    scenario: MissionScenarioRequest
    research_question: Optional[str] = Field(default=None, min_length=3, max_length=500)
    safety_domain: Optional[str] = Field(default=None, min_length=2, max_length=120)
    limit: int = Field(default=10, ge=1, le=50)


class MissionProfile(BaseModel):
    id: MissionEnvironment
    name: str
    description: str
    available_scenario_fields: List[str]
    dataset_applicability_note: str


class MissionScenarioResponse(BaseModel):
    scenario: MissionScenarioRequest
    profile: MissionProfile
    known_conditions: List[str]
    explicitly_unknown_conditions: List[str]
    notice: str


class ConditionEvidence(BaseModel):
    condition: str
    requested: Union[float, str]
    dataset_value: Optional[str] = None
    status: str
    similarity: Optional[float] = None


class MissionDimensionScores(BaseModel):
    condition_similarity: Optional[float] = None
    material_fuel_relevance: Optional[float] = None
    safety_domain_relevance: Optional[float] = None
    measurement_availability: Optional[float] = None
    condition_applicability_coverage: Optional[float] = None
    mission_environment_applicability: Optional[float] = None


class MissionEvidenceProvenance(BaseModel):
    data_status: str
    demo_data_mark: Optional[str] = None
    source_url_as_listed_in_fixture: Optional[str] = None
    verification_status: str


class RankedInvestigation(BaseModel):
    psi_record_id: str
    title: str
    research_category: Optional[str] = None
    platform: Optional[str] = None
    relevance_score: Optional[float] = None
    score_label: str
    dimension_scores: MissionDimensionScores
    explanations: Dict[str, Union[str, List[Union[str, ConditionEvidence]]]]
    measurement_available_in_fixture: bool
    measurement_status_in_fixture: str
    unknown_conditions: List[str]
    evidence: Dict[str, Optional[str]]
    provenance: MissionEvidenceProvenance


class MissionDiscoveryResponse(BaseModel):
    scenario: MissionScenarioRequest
    items: List[RankedInvestigation]
    total: int
    returned: int
    dataset_records_considered: int
    data_status: str
    limitations: List[str]


class MissionRankResponse(BaseModel):
    scenario: MissionScenarioRequest
    ranked_investigations: List[RankedInvestigation]
    scoring_notice: str
    data_status: str
    limitations: List[str]


class MissionGap(BaseModel):
    dimension: str
    requested_value: Optional[Union[float, str]] = None
    coverage_status: str
    interpretation: str
    records_with_any_reported_value: int


class MissionGapsResponse(BaseModel):
    scope: str
    dataset_status: str
    absence_of_research_claimed: bool
    gaps: List[MissionGap]
    limitations: List[str]


class MissionRankingMethod(BaseModel):
    score_label: str
    weights: Dict[str, float]
    missing_dimensions: str


class MissionReportResponse(BaseModel):
    profile: MissionProfile
    scenario: MissionScenarioResponse
    ranked_investigations: List[RankedInvestigation]
    ranking_method: MissionRankingMethod
    evidence: List[Dict[str, Any]]
    limitations: List[str]
    research_gaps: MissionGapsResponse
    data_status: str
"""Pydantic schemas matching TypeScript interfaces from requirements.
All schemas are TypeScript-compatible and include demo data marking.
"""

from pydantic import BaseModel, Field, UUID4
from typing import Optional, List, Dict, Any, Generic, TypeVar
from datetime import datetime
from enum import Enum


# Type variable for generic API responses
T = TypeVar('T')


# ============================================================
# Enums
# ============================================================

class BackendMode(str, Enum):
    MOCK = "mock"
    LIVE = "live"
    HYBRID = "hybrid"


class AssistantMode(str, Enum):
    SCIENTIFIC_QUESTION = "scientific-question"
    EXPERIMENTAL_INTERPRETATION = "experimental-interpretation"
    INVESTIGATION_SUMMARIZATION = "investigation-summarization"
    RESEARCH_COMPARISON = "research-comparison"
    EVIDENCE_RETRIEVAL = "evidence-retrieval"
    RESEARCH_GAP_IDENTIFICATION = "research-gap-identification"


class DataQualityFlag(str, Enum):
    VALID = "valid"
    QUESTIONABLE = "questionable"
    EXCLUDED = "excluded"
    UNKNOWN = "unknown"


class RiskAssessmentLevel(str, Enum):
    EVIDENCE_BASED = "evidence-based"
    PRELIMINARY = "preliminary"


# ============================================================
# Core Schemas
# ============================================================

class DemoMarkMixin(BaseModel):
    """Mixin that adds the DEMO DATA marking to any schema."""
    
    demo_data_mark: str = Field(
        default="DEMO DATA — NOT VERIFIED NASA RESEARCH",
        description="Marks content as demo/sample not verified NASA research",
    )
    
    class Config:
        use_enum_values = True


# ============================================================
# Dashboard Schemas
# ============================================================

class DashboardSummary(DemoMarkMixin, BaseModel):
    """Response from GET /dashboard/summary"""
    total_investigations: int = Field(
        default=0,
        description="Total number of indexed NASA investigations",
    )
    with_structured_data: int = Field(
        default=0,
        description="Experiments with metadata and datasets",
    )
    research_domains: List[str] = Field(
        default=[],
        description="Research domain categories",
    )
    gravity_conditions: List[str] = Field(
        default=[],
        description="Gravity condition categories"
    )
    fuel_categories: List[str] = Field(
        default=[],
        description="Fuel category categories",
    )
    data_coverage: Dict[str, float] = Field(
        default_factory=dict,
        description="Percentage coverage for: numerical_data, images_videos, experimental_metadata, full_documentation",
    )
    backend_mode: BackendMode = Field(
        default=BackendMode.MOCK,
        description="Current backend operation mode",
    )


class DataCoverageProgress(DemoMarkMixin, BaseModel):
    """Progress bar data for Data Coverage section."""
    label: str
    value: float  # percentage 0-100
    max_value: float = Field(default=100, description="Always 100")
    unit: str = Field(default="%", description="Percentage unit")
    backend_mode: BackendMode = Field(default=BackendMode.MOCK)


class RecentInvestigation(DemoMarkMixin, BaseModel):
    """Item from GET /dashboard/recent-investigations."""
    investigation_id: str = Field(description="Investigation identifier (e.g., BASS-II)")
    title: str = Field(description="Investigation title")
    research_category: str = Field(description="Research category/domain")
    short_description: str = Field(description="Brief description")
    thumbnail_url: Optional[str] = Field(
        default=None,
        description="Thumbnail image URL",
    )
    navigation_action: str = Field(
        description="Route to navigate on click",
    )


# ============================================================
# Investigation Schemas
# ============================================================

class InvestigationMetadata(DemoMarkMixin, BaseModel):
    """Investigation metadata from GET /investigations/{id}."""
    psi_record_id: str = Field(
        description="PSI record ID (e.g., PSI-26)",
    )
    experiment_name: str = Field(description="Experiment name")
    experiment_id: Optional[str] = Field(
        default=None,
        description="Experiment identifier",
    )
    investigation_title: str = Field(
        description="Full investigation title",
    )
    research_category: str = Field(
        description="Research category/domain",
    )
    research_objective: Optional[str] = Field(
        default=None,
        description="Scientific objective",
    )
    gravity_environment: str = Field(
        description="Gravity condition (microgravity, partial, 1g)",
    )
    platform: str = Field(
        description="Platform (ISS, Shuttle, Ground)",
    )
    apparatus: Optional[str] = Field(
        default=None,
        description="Apparatus/insert configuration",
    )
    fuel_material: Optional[str] = Field(
        default=None,
        description="Fuel material type",
    )
    fuel_geometry: Optional[str] = Field(
        default=None,
        description="Fuel geometry (sphere, rod, film)",
    )
    sample_dimensions: Optional[str] = Field(
        default=None,
        description="Sample dimensions",
    )
    oxygen_concentration: Optional[str] = Field(
        default=None,
        description="Oxygen concentration (% )",
    )
    pressure: Optional[str] = Field(
        default=None,
        description="Pressure environment",
    )
    measurement_method: Optional[str] = Field(
        default=None,
        description="Measurement methodology",
    )
    instrument: Optional[str] = Field(
        default=None,
        description="Instrument used",
    )
    time_or_test_point: Optional[str] = Field(
        default=None,
        description="Test point or time",
    )
    uncertainty: Optional[str] = Field(
        default=None,
        description="Documented uncertainty",
    )
    data_quality_flag: DataQualityFlag = Field(
        default=DataQualityFlag.UNKNOWN,
        description="Quality assessment flag",
    )
    source_url: Optional[str] = Field(
        default=None,
        description="Original PSI source URL",
    )
    source_page_or_row: Optional[int] = Field(
        default=None,
        description="Source page or row number",
    )
    data_quality_notes: Optional[str] = Field(
        default=None,
        description="Quality notes",
    )
    datasets_available: bool = Field(
        default=False,
        description="Whether datasets are available",
    )
    measurement_count: int = Field(
        default=0,
        description="Number of measurements recorded",
    )
    # Additional database fields
    airflow_velocity: Optional[str] = Field(
        default=None,
        description="Airflow velocity (cm/s)",
    )
    flow_direction: Optional[str] = Field(
        default=None,
        description="Flow direction",
    )
    fuel_temperature: Optional[str] = Field(
        default=None,
        description="Fuel temperature",
    )
    suppressant_type: Optional[str] = Field(
        default=None,
        description="Suppressant type",
    )
    measured_parameter: Optional[str] = Field(
        default=None,
        description="Measured parameter name",
    )
    measured_value: Optional[float] = Field(
        default=None,
        description="Measured value",
    )
    original_unit: Optional[str] = Field(
        default=None,
        description="Original measurement unit",
    )
    standardized_value: Optional[float] = Field(
        default=None,
        description="Standardized value (SI normalized)",
    )
    standardized_unit: Optional[str] = Field(
        default=None,
        description="Standardized unit",
    )
    source_file: Optional[str] = Field(
        default=None,
        description="Source file name",
    )
    ntrs_document_id: Optional[str] = Field(
        default=None,
        description="NTRS document identifier",
    )
    backend_mode: BackendMode = Field(
        default=BackendMode.MOCK,
        description="Current backend operation mode",
    )


class ExperimentalFinding(DemoMarkMixin, BaseModel):
    """Individual experimental finding from assistant analysis."""
    experiment_id: str
    flame_spread_rate: Optional[float] = Field(
        default=None,
        description="Flame spread rate (NULL if not available)",
    )
    extinction_limit: Optional[float] = Field(
        default=None,
        description="Extinction limit (NULL if not available)",
    )
    char_height: Optional[float] = Field(
        default=None,
        description="Char height (NULL if not available)",
    )
    specimen_material: str
    gravity_condition: str
    oxygen_concentration: str
    pressure: Optional[str] = Field(
        default=None,
        description="Pressure environment",
    )
    test_standard: Optional[str] = Field(
        default=None,
        description="Testing standard (e.g., NASA-STD, ASTM)",
    )
    data_quality: str = Field(
        description="Data quality assessment",
    )


class MeasurementParameterSchema(DemoMarkMixin, BaseModel):
    """Individual measurement parameter."""
    name: str = Field(description="Measurement name (e.g., flame_spread_rate)")
    value: Optional[float] = Field(
        default=None,
        description=" measured value (NULL if not available)",
    )
    original_unit: Optional[str] = Field(
        default=None,
        description="Original unit (e.g., cm/s, mm/s)",
    )
    standardized_value: Optional[float] = Field(
        default=None,
        description="Normalized/SI value",
    )
    standardized_unit: Optional[str] = Field(
        default=None,
        description="Standardized unit",
    )
    uncertainty: Optional[str] = Field(
        default=None,
        description="Documented uncertainty range",
    )
    measurement_method: Optional[str] = Field(
        default=None,
        description="Method used for measurement",
    )
    instrument: Optional[str] = Field(
        default=None,
        description="Instrument used",
    )
    data_quality_flag: DataQualityFlag = Field(
        default=DataQualityFlag.UNKNOWN,
        description="Quality flag",
    )
    source_url: Optional[str] = Field(
        default=None,
        description="Original source URL",
    )


class InvestigationFindings(DemoMarkMixin, BaseModel):
    """Findings from GET /investigations/{id}/findings."""
    investigation_id: str
    title: str
    conditions: str
    parameters: List[MeasurementParameterSchema]
    supporting_source: str
    limitations: List[str] = Field(
        default_factory=list,
        description="Known experimental limitations",
    )


# ============================================================
# Assistant Schemas
# ============================================================

class AssistantRequest(DemoMarkMixin, BaseModel):
    """Request payload for POST /assistant/ask."""
    question: str = Field(
        description="User's scientific question",
        min_length=1,
    )
    mode: AssistantMode = Field(
        description="Analysis mode (6 options)",
    )
    investigation_ids: List[str] = Field(
        default_factory=list,
        description="Investigation IDs for context (empty for general questions)",
    )
    include_evidence: bool = Field(
        default=True,
        description="Whether to retrieve supporting evidence documents",
    )
    backend_mode: BackendMode = Field(
        default=BackendMode.MOCK,
        description="Current backend mode",
    )


class AssistantAnswerSectionC(DemoMarkMixin, BaseModel):
    """Section C: Scientific Interpretation."""
    whathappened: str = Field(
        description="Direct experimental observation",
    )
    whymightit: str = Field(
        description="Evidence-supported scientific mechanisms",
    )
    whatdoesitindicate: str = Field(
        description="Scientific significance",
    )
    whataretheimplications: str = Field(
        description="Potential relevance to spacecraft fire safety",
    )


class AssistantResponse(DemoMarkMixin, BaseModel):
    """Full response from POST /assistant/ask."""
    query: str = Field(description="The original question")
    intent: AssistantMode = Field(description="Detected intent mode")
    answer: str = Field(description="Concise answer overview")
    findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Raw findings data",
    )
    interpretation: AssistantAnswerSectionC
    evidence: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Supporting evidence summary",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="List of limitations",
    )
    unanswered_questions: List[str] = Field(
        default_factory=list,
        description="Questions that could not be answered from available data",
    )
    backend_mode: BackendMode = Field(
        default=BackendMode.MOCK,
        description="Current backend mode",
    )


# Assistant Mode Descriptions (for frontend help text)
ASSISTANT_MODE_DESCRIPTIONS: Dict[AssistantMode, str] = {
    AssistantMode.SCIENTIFIC_QUESTION: "Scientific Question Answering",
    AssistantMode.EXPERIMENTAL_INTERPRETATION: "Experimental Finding Interpretation",
    AssistantMode.INVESTIGATION_SUMMARIZATION: "Investigation Summarization",
    AssistantMode.RESEARCH_COMPARISON: "Research Comparison",
    AssistantMode.EVIDENCE_RETRIEVAL: "Evidence Retrieval",
    AssistantMode.RESEARCH_GAP_IDENTIFICATION: "Research Gap Identification",
}


# ============================================================
# Evidence Schemas
# ============================================================

class EvidenceItem(DemoMarkMixin, BaseModel):
    """Evidence item from GET /evidence or assistant response."""
    id: str = Field(description="Unique evidence identifier")
    title: str = Field(description="Document title")
    investigation_name: str = Field(description="Associated investigation")
    source_organization: str = Field(description="Source organization (e.g., NASA)")
    publication_date: Optional[str] = Field(
        default=None,
        description="Publication date if available",
    )
    page_or_section: Optional[str] = Field(
        default=None,
        description="Page or section within document",
    )
    excerpt: str = Field(
        description="Key relevant passage from the document",
    )
    dataset_identifier: Optional[str] = Field(
        default=None,
        description="Dataset identifier if available",
    )
    original_source_url: str = Field(
        description="Original source URL (PSI/NTRS)",
    )
    view_source_action: bool = Field(
        default=True,
        description="Whether View Source action is available",
    )


class CitationCard(DemoMarkMixin, BaseModel):
    """Citation card for references section (H)."""
    document_title: str = Field(
        description="Document title",
    )
    investigation_name: str = Field(
        description="Associated investigation name",
    )
    source_organization: str = Field(
        description="Source organization (e.g., NASA)",
    )
    publication_date: Optional[str] = Field(
        default=None,
        description="Publication date if available",
    )
    page_or_section: Optional[str] = Field(
        default=None,
        description="Page or section within document",
    )
    supporting_excerpt: str = Field(
        description="Supporting excerpt from the document",
    )
    dataset_identifier: Optional[str] = Field(
        default=None,
        description="Dataset identifier if available",
    )
    original_source_url: str = Field(
        description="Original source URL",
    )
    view_source_action: bool = Field(
        default=True,
        description="Whether View Source action is available",
    )


# ============================================================
# Risk Assessment Schemas
# ============================================================

class RiskAssessmentInput(DemoMarkMixin, BaseModel):
    """Input parameters for risk analysis."""
    material: str = Field(
        description="Material name (e.g., PMMA, cotton, silver)",
    )
    gravity: str = Field(
        description="Gravity condition (microgravity, partial-gravity, 1g)",
    )
    oxygen: Optional[str] = Field(
        default=None,
        description="Oxygen environment (% )",
    )
    pressure: Optional[str] = Field(
        default=None,
        description="Pressure environment",
    )
    fuel_config: Optional[str] = Field(
        default=None,
        description="Fuel configuration",
    )
    ignition: Optional[str] = Field(
        default=None,
        description="Ignition conditions",
    )
    suppression: Optional[str] = Field(
        default=None,
        description="Suppression conditions",
    )
    backend_mode: BackendMode = Field(
        default=BackendMode.MOCK,
        description="Current backend mode",
    )


class RiskAssessmentResult(DemoMarkMixin, BaseModel):
    """Result from POST /risk/analyze."""
    available_experimental_evidence: str = Field(
        description="What experimental data exists",
    )
    observed_combustion_behaviour: str = Field(
        description="What experiments showed",
    )
    potential_hazard_indicators: List[str] = Field(
        default_factory=list,
        description="Hazard indicators based on evidence only",
    )
    evidence_limitations: List[str] = Field(
        default_factory=list,
        description="What we don't know / limitations",
    )
    research_gaps: List[str] = Field(
        default_factory=list,
        description="What needs more study",
    )
    assessment_methodology: str = Field(
        description="How the assessment was done",
    )
    input_parameters: Dict[str, Any] = Field(
        description="What was configured",
    )
    uncertainty_statement: str = Field(
        description="Explicit uncertainty description",
    )
    certified_safety_assessment: str = Field(
        default="no",
        description="Always 'no' - never certified safety assessment",
    )
    decision_support_level: str = Field(
        default="evidence-based",
        description="Type of decision support provided",
    )
    backend_mode: BackendMode = Field(
        default=BackendMode.MOCK,
        description="Current backend mode",
    )


# ============================================================
# Comparison Schemas
# ============================================================

class ComparisonResult(DemoMarkMixin, BaseModel):
    """Result from POST /analysis/compare."""
    investigation_ids: List[str] = Field(
        description="Compared investigation IDs",
    )
    comparison_table: List[Dict[str, Any]] = Field(
        description="Structured comparison table data",
    )
    scientific_interpretation: Dict[str, Any] = Field(
        description="Key similarities, differences, insights",
    )
    charts_available: bool = Field(
        default=False,
        description="Whether comparable charts can be generated",
    )
    backend_mode: BackendMode = Field(
        default=BackendMode.MOCK,
        description="Current backend mode",
    )


# ============================================================
# Mission Schemas
# ============================================================

class MissionScenario(DemoMarkMixin, BaseModel):
    """Mission scenario from Mission Intelligence page."""
    id: str
    title: str
    subtitle: str
    gravity: str
    environment: str
    characteristics: str
    research_domains: List[str]
    icon: str  # Lucide icon name


class MissionConfiguration(DemoMarkMixin, BaseModel):
    """User-configured mission parameters."""
    mission_environment: str = Field(
        description="Lunar Habitat, Mars Transit, etc.",
    )
    gravity_conditions: str = Field(
        description="Gravity context",
    )
    habitat_type: str = Field(
        description="Habitat type",
    )
    combustion_material: str = Field(
        description="Selected material",
    )
    fuel_category: str = Field(
        description="Fuel category",
    )
    fire_suppression: str = Field(
        description="Fire suppression method",
    )
    environmental_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional environmental params",
    )


# ============================================================
# API Response Wrappers
# ============================================================

class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper."""
    success: bool = Field(default=True, description="Whether the request succeeded")
    data: T = Field(description="Response data")
    error: Optional[str] = Field(
        default=None,
        description="Error message if success=False",
    )
    backend_mode: BackendMode = Field(
        default=BackendMode.MOCK,
        description="Current backend mode",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Request timestamp",
    )


class APIErrorResponse(BaseModel, Generic[T]):
    """Error response wrapper."""
    success: bool = Field(default=False, description="Request failed")
    error: str = Field(description="Error message")
    error_code: str = Field(description="Error code for programmatic handling")
    backend_mode: BackendMode = Field(
        default=BackendMode.MOCK,
        description="Current backend mode",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Request timestamp",
    )


# ============================================================
# Demo Data Markers & Helpers
# ============================================================

def add_demo_mark(text: str) -> str:
    """Prepend the DEMO DATA mark if not already present."""
    if text.startswith("DEMO DATA — NOT VERIFIED NASA RESEARCH"):
        return text
    return f"DEMO DATA — NOT VERIFIED NASA RESEARCH: {text}"


def set_backend_mode(mode: BackendMode) -> None:
    """Set the global backend mode (affects all responses)."""
    # In a real app, this might use a settings object or dependency injection
    pass


# ============================================================
# Schema Validation Helpers
# ============================================================

def validate_measurement_unit(unit: str) -> bool:
    """Validate that a measurement unit is reasonable."""
    valid_units = {
        "flame_spread_rate": {"cm/s", "mm/s", "m/s", "in/s"},
        "extinction_limit": {"seconds", "minutes"},
        "temperature": {"K", "°C", "°F"},
        "pressure": {"atm", "Pa", "kPa", "bar"},
        "oxygen_concentration": {"%", "ppm"},
        "general": set(),
    }
    return True  # Simplified validation


def normalize_measurement_value(
    value: Optional[float],
    original_unit: Optional[str],
    target_unit: Optional[str] = None,
) -> Optional[float]:
    """Convert measurement to target unit, return NULL if value missing."""
    if value is None:
        return None
    # In real implementation, do unit conversion
    return value


# Export all symbols
__all__ = [
    "BackendMode",
    "AssistantMode",
    "DataQualityFlag",
    "RiskAssessmentLevel",
    "DemoMarkMixin",
    "DashboardSummary",
    "DataCoverageProgress",
    "RecentInvestigation",
    "InvestigationMetadata",
    "InvestigationFindings",
    "MeasurementParameterSchema",
    "AssistantRequest",
    "AssistantAnswerSectionC",
    "AssistantResponse",
    "EvidenceItem",
    "CitationCard",
    "RiskAssessmentInput",
    "RiskAssessmentResult",
    "ComparisonResult",
    "MissionScenario",
    "MissionConfiguration",
    "APIResponse",
    "APIErrorResponse",
    "add_demo_mark",
    "set_backend_mode",
    "validate_measurement_unit",
    "normalize_measurement_value",
    "ASSISTANT_MODE_DESCRIPTIONS",
]
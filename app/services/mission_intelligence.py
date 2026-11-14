"""Dataset-grounded mission relevance logic; scores are not safety predictions."""

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.models.mission_intelligence import (
    MissionEnvironment,
    MissionProfile,
    MissionQueryRequest,
    MissionScenarioRequest,
)


DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"
INVESTIGATIONS_CSV = DATA_ROOT / "psi_investigations_dataset.csv"
FULL_DATASET_CSV = DATA_ROOT / "psi_full_dataset.csv"
DATA_STATUS = "DEMO_DATA_NOT_VERIFIED_NASA_RESEARCH"
UNKNOWN_MARKERS = {"", "n/a", "na", "n.a.", "not available", "unknown", "none", "null", "variable"}
LIMITATIONS = [
    "The provided CSVs are explicitly marked demo data and have not been verified against NASA PSI.",
    "Scores order research relevance heuristically; they are not validated fire-risk, hazard, or safety scores.",
    "The fixtures contain only ISS/microgravity records and do not establish lunar, Martian, rover, or habitat validation.",
    "Blank, N/A, and variable experimental conditions remain unknown; they are not treated as zero or as negative findings.",
    "The full fixture has no reported measured_parameter or measured_value entries, so measurement availability is not demonstrated.",
]

PROFILE_DETAILS = {
    MissionEnvironment.SPACECRAFT: (
        "Spacecraft mission context.",
        "The fixture labels some platforms as ISS and includes spacecraft-related descriptions; exact mission applicability is not validated.",
    ),
    MissionEnvironment.SPACE_STATION: (
        "Space station mission context.",
        "Fixture platform labels include ISS, but records are demo-marked and are not independently verified.",
    ),
    MissionEnvironment.LUNAR_HABITAT: (
        "Lunar habitat mission context.",
        "No lunar-habitat field or partial-gravity result is established by these fixtures.",
    ),
    MissionEnvironment.MARS_HABITAT: (
        "Mars habitat mission context.",
        "No Mars-habitat field or partial-gravity result is established by these fixtures.",
    ),
    MissionEnvironment.PRESSURIZED_ROVER: (
        "Pressurized rover mission context.",
        "No rover-specific experimental context is established by these fixtures.",
    ),
}


class DatasetLoadError(RuntimeError):
    """Raised when required dataset files cannot be read or parsed."""


def _read_csv(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, strict=True)
            if not reader.fieldnames or "psi_record_id" not in reader.fieldnames:
                raise DatasetLoadError("Required mission-intelligence dataset has no psi_record_id column.")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise DatasetLoadError("Required mission-intelligence dataset contains duplicate column names.")
            rows = []
            for row in reader:
                if None in row:
                    raise DatasetLoadError("Required mission-intelligence dataset contains a malformed row.")
                rows.append(row)
            return rows
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetLoadError("Required mission-intelligence dataset is unavailable or invalid.") from exc


@lru_cache(maxsize=1)
def load_investigations() -> Tuple[Dict[str, str], ...]:
    """Merge fixture records by their actual psi_record_id without inventing fields."""
    investigation_rows = _read_csv(INVESTIGATIONS_CSV)
    full_rows = _read_csv(FULL_DATASET_CSV)
    merged: Dict[str, Dict[str, str]] = {}
    for row in full_rows:
        record_id = _known(row.get("psi_record_id")) or ""
        if record_id:
            merged[record_id] = dict(row)
    for row in investigation_rows:
        record_id = _known(row.get("psi_record_id")) or ""
        if record_id:
            existing = merged.get(record_id, {})
            merged[record_id] = {
                **existing,
                **{
                    key: value
                    for key, value in row.items()
                    if key not in existing or _known(value) is not None
                },
            }
    return tuple(merged[key] for key in sorted(merged))


def _known(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return None if cleaned.lower() in UNKNOWN_MARKERS else cleaned


def _numeric(value: Optional[str], field: str) -> Optional[Tuple[float, float]]:
    """Parse a reported scalar/range to a normalized range, returning None if unknown."""
    cleaned = _known(value)
    if cleaned is None:
        return None
    # Scenario values are constrained nonnegative; don't treat a range separator (10-30)
    # as a negative sign on the second endpoint.
    matches = re.findall(r"(?:\d+(?:\.\d*)?|\.\d+)", cleaned.replace(",", ""))
    if not matches:
        return None
    numbers = [float(item) for item in matches[:2]]
    if field == "oxygen":
        if "%" not in cleaned:
            return None
    elif field == "pressure":
        unit_match = re.search(r"(?<![a-z])(?:kpa|pa|atm|bar)(?![a-z])", cleaned.lower())
        if not unit_match:
            return None
        unit = unit_match.group(0)
        factor = {"kpa": 1.0, "pa": 0.001, "atm": 101.325, "bar": 100.0}[unit]
        numbers = [number * factor for number in numbers]
    elif field == "airflow":
        unit_match = re.search(r"(?<![a-z])(?:km/h|cm/s|mm/s|m/s)(?![a-z])", cleaned.lower())
        if not unit_match:
            return None
        factor = {"m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001, "km/h": 1.0 / 3.6}[unit_match.group(0)]
        numbers = [number * factor for number in numbers]
    if len(numbers) == 1:
        numbers.append(numbers[0])
    return (min(numbers), max(numbers))


def _field_value(row: Dict[str, str], key: str) -> Optional[str]:
    return _known(row.get(key))


def get_profiles() -> List[Dict[str, Any]]:
    return [
        MissionProfile(
            id=environment,
            name=environment.value.replace("_", " ").title(),
            description=PROFILE_DETAILS[environment][0],
            available_scenario_fields=[
                "oxygen_concentration_pct", "pressure_kpa", "gravity_condition",
                "airflow_velocity_m_s", "habitat_type", "fuel_material",
            ],
            dataset_applicability_note=PROFILE_DETAILS[environment][1],
        ).model_dump(mode="json")
        for environment in MissionEnvironment
    ]


def scenario_summary(scenario: MissionScenarioRequest) -> Dict[str, Any]:
    profile = next(item for item in get_profiles() if item["id"] == scenario.mission_environment.value)
    condition_fields = {
        "oxygen_concentration_pct": scenario.oxygen_concentration_pct,
        "pressure_kpa": scenario.pressure_kpa,
        "gravity_condition": scenario.gravity_condition,
        "airflow_velocity_m_s": scenario.airflow_velocity_m_s,
        "habitat_type": scenario.habitat_type,
        "fuel_material": scenario.fuel_material,
    }
    known = [key for key, value in condition_fields.items() if value is not None]
    unknown = [key for key, value in condition_fields.items() if value is None]
    return {
        "scenario": scenario.model_dump(mode="json"),
        "profile": profile,
        "known_conditions": known,
        "explicitly_unknown_conditions": unknown,
        "notice": "Unspecified conditions remain unknown and are not filled from assumptions or mission defaults.",
    }


def _condition_evidence(row: Dict[str, str], scenario: MissionScenarioRequest) -> Tuple[Optional[float], List[Dict[str, Any]], int, int]:
    requests = [
        ("oxygen_concentration_pct", scenario.oxygen_concentration_pct, "oxygen_concentration", "oxygen"),
        ("pressure_kpa", scenario.pressure_kpa, "pressure", "pressure"),
        ("airflow_velocity_m_s", scenario.airflow_velocity_m_s, "airflow_velocity", "airflow"),
    ]
    scores: List[float] = []
    evidence: List[Dict[str, Any]] = []
    requested_count = 0
    reported_count = 0
    for request_name, requested, column, kind in requests:
        if requested is None:
            continue
        requested_count += 1
        raw = row.get(column)
        parsed = _numeric(raw, kind)
        if parsed is None:
            evidence.append({"condition": request_name, "requested": requested, "dataset_value": _known(raw), "status": "unknown_in_fixture"})
            continue
        reported_count += 1
        low, high = parsed
        if low <= requested <= high:
            similarity = 1.0
            status = "within_reported_range"
        else:
            nearest = low if requested < low else high
            scale = 100.0 if kind == "oxygen" else max(abs(requested), abs(nearest), 1.0)
            similarity = max(0.0, 1.0 - abs(requested - nearest) / scale)
            status = "outside_reported_range"
        scores.append(similarity)
        evidence.append({"condition": request_name, "requested": requested, "dataset_value": _known(raw), "status": status, "similarity": round(similarity, 4)})

    if scenario.gravity_condition is not None:
        requested_count += 1
        raw_gravity = row.get("gravity_environment")
        known_gravity = _known(raw_gravity)
        if known_gravity is None:
            evidence.append({"condition": "gravity_condition", "requested": scenario.gravity_condition, "dataset_value": None, "status": "unknown_in_fixture"})
        else:
            reported_count += 1
            similarity = 1.0 if known_gravity.casefold() == scenario.gravity_condition.casefold() else 0.0
            scores.append(similarity)
            evidence.append({"condition": "gravity_condition", "requested": scenario.gravity_condition, "dataset_value": known_gravity, "status": "match" if similarity else "different_reported_condition", "similarity": similarity})

    similarity_score = sum(scores) / len(scores) if scores else None
    return similarity_score, evidence, reported_count, requested_count


def _word_set(text: Optional[str]) -> set:
    if not text:
        return set()
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2}


def _score_record(row: Dict[str, str], request: MissionQueryRequest) -> Dict[str, Any]:
    scenario = request.scenario
    condition_score, condition_evidence, reported_conditions, requested_conditions = _condition_evidence(row, scenario)
    target_material = scenario.fuel_material
    record_material = _field_value(row, "fuel_material")
    if target_material:
        if record_material is None:
            material_score: Optional[float] = None
            material_note = "Fuel/material is unknown in this fixture record."
        else:
            material_score = 1.0 if target_material.casefold() in record_material.casefold() or record_material.casefold() in target_material.casefold() else 0.0
            material_note = "Textual material match in fixture fields." if material_score else "No textual material match in the reported fixture fields."
    else:
        material_score = None
        material_note = "No target material was supplied; material relevance was not assessed."

    desired_domain = request.safety_domain or request.research_question
    source_text = " ".join(_field_value(row, key) or "" for key in ("research_category", "research_objective", "investigation_title"))
    if desired_domain:
        desired_words = _word_set(desired_domain)
        source_words = _word_set(source_text)
        domain_score: Optional[float] = len(desired_words & source_words) / len(desired_words) if desired_words else None
        domain_note = "Lexical overlap only; this does not establish operational safety applicability."
    else:
        domain_score = None
        domain_note = "No safety domain or research question was supplied; domain relevance was not assessed."

    measured_parameter = _field_value(row, "measured_parameter")
    measured_value = _field_value(row, "measured_value")
    measurement_available = bool(measured_parameter and measured_value and _numeric(measured_value, "measurement"))
    # Missing fixture values are unknown availability, not evidence of no measurements.
    measurement_score = 1.0 if measurement_available else None
    measurement_note = (
        "A measured parameter and numeric value are present in the fixture."
        if measurement_available
        else "A measured parameter/value pair is not reported in the full fixture; availability is unknown, not zero or proof of absent research."
    )

    applicability_score = reported_conditions / requested_conditions if requested_conditions else None
    dimensions: Dict[str, Optional[float]] = {
        "condition_similarity": condition_score,
        "material_fuel_relevance": material_score,
        "safety_domain_relevance": domain_score,
        "measurement_availability": measurement_score,
        "condition_applicability_coverage": applicability_score,
        "mission_environment_applicability": None,
    }
    weights = {
        "condition_similarity": 0.27,
        "material_fuel_relevance": 0.23,
        "safety_domain_relevance": 0.14,
        "measurement_availability": 0.14,
        "condition_applicability_coverage": 0.12,
        "mission_environment_applicability": 0.10,
    }
    included_weight = sum(weights[key] for key, value in dimensions.items() if value is not None)
    score = (
        100.0 * sum(weights[key] * value for key, value in dimensions.items() if value is not None) / included_weight
        if included_weight else None
    )
    record_id = row.get("psi_record_id") or ""
    raw_evidence = {
        key: _known(row.get(key))
        for key in (
            "experiment_name", "investigation_title", "research_category", "research_objective",
            "gravity_environment", "platform", "fuel_material", "fuel_geometry",
            "sample_dimensions", "oxygen_concentration", "pressure", "airflow_velocity",
            "measurement_method", "instrument", "measured_parameter", "measured_value",
            "source_url", "source_page_or_row", "datasets_available", "measurement_count",
        )
    }
    explanations = {
        "condition_similarity": condition_evidence or ["No scenario conditions were supplied for direct comparison."],
        "material_fuel_relevance": material_note,
        "safety_domain_relevance": domain_note,
        "measurement_availability": measurement_note,
        "condition_applicability_coverage": (
            "{} of {} requested conditions have a reported value in this fixture.".format(reported_conditions, requested_conditions)
            if requested_conditions else "No comparable physical conditions were supplied."
        ),
        "mission_environment_applicability": "Not scored: the fixtures do not establish specific habitat, rover, lunar, or Mars applicability.",
    }
    return {
        "psi_record_id": record_id,
        "title": _field_value(row, "investigation_title") or _field_value(row, "experiment_name") or record_id,
        "research_category": _field_value(row, "research_category"),
        "platform": _field_value(row, "platform"),
        "relevance_score": round(score, 2) if score is not None else None,
        "score_label": "HEURISTIC_RESEARCH_RELEVANCE_NOT_FIRE_RISK_OR_SAFETY_SCORE",
        "dimension_scores": {key: round(value, 4) if value is not None else None for key, value in dimensions.items()},
        "explanations": explanations,
        "measurement_available_in_fixture": measurement_available,
        "measurement_status_in_fixture": "reported" if measurement_available else "not_reported_unknown",
        "unknown_conditions": [item["condition"] for item in condition_evidence if item["status"] == "unknown_in_fixture"],
        "evidence": raw_evidence,
        "provenance": {
            "data_status": DATA_STATUS,
            "demo_data_mark": _field_value(row, "demo_data_mark"),
            "source_url_as_listed_in_fixture": _field_value(row, "source_url"),
            "verification_status": "not independently verified",
        },
    }


def rank_investigations(request: MissionQueryRequest) -> List[Dict[str, Any]]:
    ranked = [_score_record(row, request) for row in load_investigations()]
    return sorted(
        ranked,
        key=lambda item: (
            item["relevance_score"] is None,
            -(item["relevance_score"] or 0.0),
            item["psi_record_id"],
        ),
    )[:request.limit]


def discover_investigations(request: MissionQueryRequest) -> Dict[str, Any]:
    items = rank_investigations(request)
    return {
        "scenario": request.scenario.model_dump(mode="json"),
        "items": items,
        "total": len(load_investigations()),
        "returned": len(items),
        "dataset_records_considered": len(load_investigations()),
        "data_status": DATA_STATUS,
        "limitations": LIMITATIONS,
    }


def identify_gaps(scenario: MissionScenarioRequest) -> Dict[str, Any]:
    rows = load_investigations()
    dimensions: List[Tuple[str, Optional[str], str]] = [
        ("gravity_condition", scenario.gravity_condition, "gravity_environment"),
        ("oxygen_concentration_pct", str(scenario.oxygen_concentration_pct) if scenario.oxygen_concentration_pct is not None else None, "oxygen_concentration"),
        ("pressure_kpa", str(scenario.pressure_kpa) if scenario.pressure_kpa is not None else None, "pressure"),
        ("airflow_velocity_m_s", str(scenario.airflow_velocity_m_s) if scenario.airflow_velocity_m_s is not None else None, "airflow_velocity"),
        ("fuel_material", scenario.fuel_material, "fuel_material"),
        ("habitat_type", scenario.habitat_type, "habitat_type"),
    ]
    gaps: List[Dict[str, Any]] = []
    for dimension, requested, column in dimensions:
        reported_rows = [row for row in rows if _known(row.get(column)) is not None]
        if dimension in ("oxygen_concentration_pct", "pressure_kpa", "airflow_velocity_m_s") and requested is not None:
            kind = {"oxygen_concentration_pct": "oxygen", "pressure_kpa": "pressure", "airflow_velocity_m_s": "airflow"}[dimension]
            number = float(requested)
            matching_ids = [
                row.get("psi_record_id") for row in reported_rows
                if (values := _numeric(row.get(column), kind)) is not None and values[0] <= number <= values[1]
            ]
        elif requested is not None:
            matching_ids = [row.get("psi_record_id") for row in reported_rows if requested.casefold() in (_known(row.get(column)) or "").casefold()]
        else:
            matching_ids = []
        if requested is not None and not matching_ids:
            gaps.append({
                "dimension": dimension,
                "requested_value": requested,
                "coverage_status": "not_found_in_provided_csv_fixtures",
                "interpretation": "Dataset coverage gap only; this does not demonstrate absence of NASA or other research.",
                "records_with_any_reported_value": len(reported_rows),
            })

    gaps.append({
        "dimension": "mission_environment_applicability",
        "requested_value": scenario.mission_environment.value,
        "coverage_status": "not_directly_represented_in_provided_csv_fixtures",
        "interpretation": "No CSV field directly encodes this mission environment. Platform/context mentions are not mission-environment validation; this coverage gap does not demonstrate absence of NASA or other research.",
        "records_with_any_reported_value": 0,
    })
    measurement_count = sum(
        1 for row in rows
        if _field_value(row, "measured_parameter") and _field_value(row, "measured_value") and _numeric(row.get("measured_value"), "measurement")
    )
    gaps.append({
        "dimension": "numeric_measurement_values",
        "requested_value": None,
        "coverage_status": "no_numeric_measurement_pairs_in_provided_full_csv_fixture" if measurement_count == 0 else "reported_in_fixture",
        "interpretation": "Dataset coverage gap only; this does not demonstrate absence of NASA or other research or measurements elsewhere.",
        "records_with_any_reported_value": measurement_count,
    })
    return {
        "scope": "provided_csv_fixtures_only",
        "dataset_status": DATA_STATUS,
        "absence_of_research_claimed": False,
        "gaps": gaps,
        "limitations": LIMITATIONS,
    }


def build_report(request: MissionQueryRequest) -> Dict[str, Any]:
    ranked = rank_investigations(request)
    return {
        "profile": next(item for item in get_profiles() if item["id"] == request.scenario.mission_environment.value),
        "scenario": scenario_summary(request.scenario),
        "ranked_investigations": ranked,
        "ranking_method": {
            "score_label": "HEURISTIC_RESEARCH_RELEVANCE_NOT_FIRE_RISK_OR_SAFETY_SCORE",
            "weights": {
                "condition_similarity": 0.27,
                "material_fuel_relevance": 0.23,
                "safety_domain_relevance": 0.14,
                "measurement_availability": 0.14,
                "condition_applicability_coverage": 0.12,
                "mission_environment_applicability": 0.10,
            },
            "missing_dimensions": "Excluded and remaining weights renormalized; unknown conditions are never assumed or treated as zero measurements.",
        },
        "evidence": [item["provenance"] | {"psi_record_id": item["psi_record_id"], "fields": item["evidence"]} for item in ranked],
        "limitations": LIMITATIONS,
        "research_gaps": identify_gaps(request.scenario),
        "data_status": DATA_STATUS,
    }
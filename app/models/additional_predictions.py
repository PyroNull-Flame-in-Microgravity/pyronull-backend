import math
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Psi10InverseFlameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    oxygen_mole_fraction: float = Field(ge=0, le=1, allow_inf_nan=False, description="XO2, oxygen mole fraction.")
    oxygen_mass_flow_mg_s: float = Field(ge=0, allow_inf_nan=False, description="mO2 in mg/s.")
    nitrogen_mass_flow_mg_s: float = Field(ge=0, allow_inf_nan=False, description="mN2 in mg/s.")
    total_mass_flow_mg_s: float = Field(ge=0, allow_inf_nan=False, description="mtot in mg/s.")
    initial_pressure_atm: float = Field(gt=0, allow_inf_nan=False, description="pinit in atm.")
    mean_pressure_atm: float = Field(gt=0, allow_inf_nan=False, description="pmean in atm.")

    @field_validator(
        "oxygen_mole_fraction", "oxygen_mass_flow_mg_s", "nitrogen_mass_flow_mg_s",
        "total_mass_flow_mg_s", "initial_pressure_atm", "mean_pressure_atm", mode="before",
    )
    @classmethod
    def reject_non_finite_numbers(cls, value):
        if isinstance(value, (int, float)) and not math.isfinite(value):
            return repr(value)
        return value


class Psi159IgnitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flame_type: str = Field(min_length=1, max_length=80, description="Flame Type as used in PSI-159 CFI-G data.")
    fuel: str = Field(min_length=1, max_length=120, description="Fuel label as used in PSI-159 CFI-G data.")
    pressure_bar: float = Field(gt=0, allow_inf_nan=False, description="Approximate pressure in bar.")
    adiabatic_flame_temperature_k: float = Field(gt=0, allow_inf_nan=False, description="Tad in kelvin.")
    hydrocarbon_burn_rate_mg_s: float = Field(ge=0, allow_inf_nan=False, description="Hydrocarbon burn rate in mg/s.")
    oxygen_flow_rate_mg_s: float = Field(ge=0, allow_inf_nan=False, description="Oxygen flow rate in mg/s.")
    oxygen_mole_fraction: float = Field(ge=0, le=1, allow_inf_nan=False, description="XO2 mole fraction.")
    fuel_fraction: float = Field(ge=0, le=1, allow_inf_nan=False, description="Xf fuel fraction.")

    @field_validator("flame_type", "fuel")
    @classmethod
    def non_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value must not be blank.")
        return value

    @field_validator(
        "pressure_bar", "adiabatic_flame_temperature_k", "hydrocarbon_burn_rate_mg_s",
        "oxygen_flow_rate_mg_s", "oxygen_mole_fraction", "fuel_fraction", mode="before",
    )
    @classmethod
    def reject_non_finite_numbers(cls, value):
        if isinstance(value, (int, float)) and not math.isfinite(value):
            return repr(value)
        return value


class PredictionEvaluationMetrics(BaseModel):
    mae: Optional[float] = None
    mae_unit: Optional[str] = None
    rmse: Optional[float] = None
    rmse_unit: Optional[str] = None
    r_squared: Optional[float] = None
    accuracy: Optional[float] = None
    cross_validation_accuracy: Optional[float] = None
    cross_validation_standard_deviation: Optional[float] = None
    documented_sample_count: Optional[int] = None
    documented_test_split_ratio: Optional[float] = None
    sample_count_note: Optional[str] = None


class PredictionInputValidation(BaseModel):
    status: str
    feature_order: List[str]
    units: Dict[str, str]
    note: str


class PredictionProvenance(BaseModel):
    dataset_filename: str
    source_url: Optional[str] = None
    verification_status: str = "UNVERIFIED"
    note: str


class Psi10RegressionPredictionResponse(BaseModel):
    investigation_id: str = "PSI-10"
    model_name: str
    prediction_target: str
    predicted_value: float
    output_unit: str
    evaluation_metrics: PredictionEvaluationMetrics
    model_status: str = "AVAILABLE"
    input_validation: PredictionInputValidation
    provenance: PredictionProvenance
    limitations: List[str]
    scientific_warning: str


class Psi159IgnitionPredictionResponse(BaseModel):
    investigation_id: str = "PSI-159"
    model_name: str
    prediction_target: str = "Hot ignition / flash classification"
    predicted_class: str
    predicted_class_code: int
    output_unit: Optional[str] = None
    evaluation_metrics: PredictionEvaluationMetrics
    model_status: str = "AVAILABLE"
    input_validation: PredictionInputValidation
    provenance: PredictionProvenance
    limitations: List[str]
    scientific_warning: str


class Psi107FuelClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nozzle_mm: float = Field(gt=0, allow_inf_nan=False, description="Nozzle diameter in mm.")
    fuel_rate_mg_s: float = Field(ge=0, allow_inf_nan=False, description="Fuel rate in mg/s.")
    hydrocarbon_fuel_rate_mg_s: float = Field(ge=0, allow_inf_nan=False, description="Hydrocarbon fuel rate in mg/s.")
    air_rate_cm_s: float = Field(allow_inf_nan=False, description="Air rate in cm/s; the source data includes negative values, so no sign restriction is applied.")

    @field_validator("nozzle_mm", "fuel_rate_mg_s", "hydrocarbon_fuel_rate_mg_s", "air_rate_cm_s", mode="before")
    @classmethod
    def reject_non_finite_numbers(cls, value):
        if isinstance(value, (int, float)) and not math.isfinite(value):
            return repr(value)
        return value


class Psi69FlameOutcomeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ambient_pressure_mmhg: float = Field(ge=0, allow_inf_nan=False, description="Ambient pressure in mmHg.")
    oxygen_mole_fraction: float = Field(ge=0, le=1, allow_inf_nan=False, description="Initial O mole fraction, matching the serialized preprocessor feature name.")
    nitrogen_mole_fraction: float = Field(ge=0, le=1, allow_inf_nan=False, description="Initial N mole fraction, matching the serialized preprocessor feature name.")
    carbon_monoxide_mole_fraction: float = Field(ge=0, le=1, allow_inf_nan=False, description="Initial CO mole fraction, matching the serialized preprocessor feature name.")
    helium_mole_fraction: float = Field(ge=0, le=1, allow_inf_nan=False, description="Initial He mole fraction.")
    droplet_initial_diameter_mm: float = Field(gt=0, allow_inf_nan=False, description="Initial droplet diameter in mm.")
    fuel: Literal["Methanol", "Heptane"] = Field(description="Fuel category present in PSI-69 dataset.")

    @field_validator(
        "ambient_pressure_mmhg", "oxygen_mole_fraction", "nitrogen_mole_fraction",
        "carbon_monoxide_mole_fraction", "helium_mole_fraction", "droplet_initial_diameter_mm",
        mode="before",
    )
    @classmethod
    def reject_non_finite_numbers(cls, value):
        if isinstance(value, (int, float)) and not math.isfinite(value):
            return repr(value)
        return value


class AdditionalClassificationResponse(BaseModel):
    investigation_id: str
    model_name: str
    prediction_target: str
    predicted_class: str
    predicted_class_index: int
    output_unit: Optional[str] = None
    evaluation_metrics: PredictionEvaluationMetrics
    model_status: str = "AVAILABLE"
    input_validation: PredictionInputValidation
    provenance: PredictionProvenance
    limitations: List[str]
    scientific_warning: str

from typing import Dict, List
import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BurnTimePredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flame_type: str = Field(min_length=1, max_length=100, description="Experimental flame type, e.g. Normal or Inverse.")
    fuel: str = Field(min_length=1, max_length=120, description="Fuel label as represented in the PSI-159 training data.")
    pressure_bar: float = Field(gt=0, allow_inf_nan=False, description="Approximate pressure in bar.")
    adiabatic_flame_temperature_k: float = Field(gt=0, allow_inf_nan=False, description="Adiabatic flame temperature in kelvin.")
    hydrocarbon_burn_rate_mg_s: float = Field(ge=0, allow_inf_nan=False, description="Hydrocarbon burn rate in mg/s.")
    oxygen_flow_rate_mg_s: float = Field(ge=0, allow_inf_nan=False, description="Oxygen flow rate in mg/s.")
    oxygen_mole_fraction: float = Field(ge=0, le=1, allow_inf_nan=False, description="Oxygen mole fraction (XO2).")
    fuel_fraction: float = Field(ge=0, le=1, allow_inf_nan=False, description="Fuel fraction (Xf).")

    @field_validator(
        "pressure_bar",
        "adiabatic_flame_temperature_k",
        "hydrocarbon_burn_rate_mg_s",
        "oxygen_flow_rate_mg_s",
        "oxygen_mole_fraction",
        "fuel_fraction",
        mode="before",
    )
    @classmethod
    def reject_non_finite_numbers(cls, value):
        if isinstance(value, (int, float)) and not math.isfinite(value):
            # Keep validation error input JSON-serializable (FastAPI echoes it in 422 details).
            return repr(value)
        return value

    @field_validator("flame_type", "fuel")
    @classmethod
    def non_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value must not be blank.")
        return value


class BurnTimeEvaluationMetrics(BaseModel):
    mae_seconds: float = Field(description="Mean absolute error on the documented evaluation set, in seconds.")
    rmse_seconds: float = Field(description="Root mean squared error on the documented evaluation set, in seconds.")
    r_squared: float = Field(description="R-squared on the documented evaluation set.")


class BurnTimePredictionResponse(BaseModel):
    predicted_burn_time_seconds: float
    model_name: str
    model_type: str
    evaluation_metrics: BurnTimeEvaluationMetrics
    model_limitations: List[str]
    experimental_domain_warning: str

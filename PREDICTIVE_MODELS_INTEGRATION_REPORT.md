# Predictive Models Integration Report

**Generated:** 2026-11-15
**Source archive:** `Predictive-Models-main (1).zip`
**Archive SHA-256:** `04215418e96fd2af1fbc7769091c67fbbab85dcc1b4c4d5c7eb4b818fd89e729`
**NASA verification:** No model, source URL, or dataset in the archive was independently confirmed against NASA. All source-repository models are `UNVERIFIED`.

## Executive summary

The archive contains **eight candidate models** across ten serialized files: three PSI-10 regressors, one PSI-107 fuel classifier, two PSI-159 models, and two PSI-69 DNNs (each PSI-69 DNN has a Keras model plus a preprocessing artifact). **Seven model endpoints are available**: three PSI-10, one PSI-107, two PSI-159, and one PSI-69. The PSI-69/01 regression candidate remains blocked because its target and output unit cannot be identified.

The PSI-107 and PSI-69/02 models were deserialized only after inspecting documentation and static artifact structure, then exercised with the example inputs in the archive. PSI-69/02's stored model config uses Keras 3.13.2 while the Python 3.9 macOS-arm64 environment uses Keras 3.10.0; a narrow in-memory removal of a nullable `quantization_config` key permits loading the unchanged architecture and weights. The `.keras` artifact itself is integrity-checked and never rewritten.

## Complete archive inventory

| Investigation/model | Archive artifacts | Dataset(s) | Target / output | Inputs and preprocessing | Metrics and counts | Status |
|---|---|---|---|---|---|---|
| PSI-10 `01-Model` inverse-flame Tad | `inverse_flame_tad_predictor.joblib`; README, JSON and HTML cards | Inverse flame CSV, 120 rows; normal flame CSV, 207 rows | `Tad (K)` | Six numeric inputs, in order: `XO2` (fraction), `mO2 (mg/s)`, `mN2 (mg/s)`, `mtot (mg/s)`, `pinit (atm)`, `pmean (atm)`. RandomForestRegressor, 100 estimators; no separate preprocessing documented. Artifact static metadata says scikit-learn 1.6.1 and feature names agree. | Card: 87 samples, 20% test split, MAE 53.50 K, R² 0.6218. The 87/120 sample discrepancy is unexplained. | **Integrated, UNVERIFIED** |
| PSI-10 `02-Model` inverse-flame mean Zst | `inverse_flame_zst_predictor.joblib`; README and JSON card | Same PSI-10 CSVs | `mean Zst`; unit not explicitly stated (dimensionless is inferred from target name) | Same six inputs and RandomForest configuration; static artifact feature names agree; scikit-learn 1.6.1. | 87 samples, 20% split, MAE 0.024039, R² 0.9305; row selection discrepancy unexplained. | **Integrated, UNVERIFIED** |
| PSI-10 `03-Model` pressure rise | `inverse_flame_pressure_rise_predictor.joblib`; README and JSON card | Same PSI-10 CSVs | `pfinal - pinit`, atm | Same six inputs and RandomForest configuration; static artifact feature names agree; scikit-learn 1.6.1. | 87 samples, 20% split, MAE 0.001406 atm, R² 0.6030; row selection discrepancy unexplained. | **Integrated, UNVERIFIED** |
| PSI-107 fuel classifier | `fuel_classifier_pipeline.joblib`; README and JSON card | `PSI-107_Experimental table_Spice.csv`, 469 rows | Fuel class, categorical output | Inputs: Nozzle (mm), Fuel Rate (mg/s), HC Fuel Rate (mg/s), Air Rate (cm/s). README architecture: 4→64→32→6 MLP, ReLU/dropout 0.2. Packet contains `model_state_dict`, fitted `StandardScaler`, ordered `classes`, and `input_features`. Runtime code reconstructs the documented architecture and class ordering from packet metadata. PyTorch 2.6.0; sklearn scaler artifact records 1.6.1. | Card accuracy 0.8191; split and counts not documented. | **Integrated, UNVERIFIED** |
| PSI-159 ignition classifier | `PSI_159_ignition_classifier_Predictive_Model.joblib`; README | Inverse-flame CSV, 1,015 rows; normal-flame CSV, 179 rows | No ignition vs. Hot Ignition/Flash, class code 0/1 | Eight inputs: Flame Type, Fuel, approx p (bar), Tad (K), HC burnrate (mg/s), O2 flow rate (mg/s), XO2, Xf. Pipeline includes median/mode imputation and one-hot categorical encoding; metadata records scikit-learn 1.6.1. | README test accuracy 90.48%; five-fold CV 88.88% ± 2.83%; sample/split details unavailable. | **Integrated, UNVERIFIED** |
| PSI-159 burn-time regression | `PSI_159_burntime_regressor_Predictive_Model.joblib`; README | Same PSI-159 tables | Burn time (sec) | Same eight inputs; preprocessing pipeline; scikit-learn 1.6.1. Existing implementation preserved. | MAE 23.08 sec, RMSE 36.14 sec, R² 0.1575; sample/split details unavailable. | **Already integrated, UNVERIFIED** |
| PSI-69 `01-Model` scalar DNN | `dnn_model.keras`, `dnn_preprocessor.joblib`; README/model card are empty | FLEX CSV, 274 rows | **Unknown.** Keras archive is a regression shape (8 inputs, one linear output, MSE), but no evidence identifies the target or output unit. Candidate dataset targets include extinction diameter, burning rate, and burn time. | Preprocessor static feature names: ambient pressure (mmHg), initial O/N/CO/He fractions, droplet initial diameter (mm), Fuel; median imputation, scaling, one-hot encoding. Keras metadata says 3.13.2. | No metrics, split, or training mapping documented. | **BLOCKED** — target/output cannot be identified; no endpoint exposed. |
| PSI-69 `02-Model` flame-outcome classifier | `dnn_cls_model.keras`, `dnn_cls_preprocessor.joblib`; README and model card | FLEX CSV, 274 rows | `Test end`: Completion, Disruption, Extinction | Seven raw inputs: pressure, initial O/N/CO/He fractions, droplet diameter, Fuel; serialized preprocessor records exact O/N/CO names and produces 8 transformed columns. Keras config is 8→64→32→3 softmax with BatchNorm/dropout. Keras archive declares version 3.13.2. Class label ordering used by its README is not backed by training source. | Card: 274 samples, 80/20 split, SMOTE training balanced to 447; no numerical test metric. | **Integrated, UNVERIFIED; class ordering caveat** |

### Dataset inventory

| Dataset | Rows | Notes |
|---|---:|---|
| `PSI-10_Experimental table_Flame Design_Inverse Flame.csv` | 120 | All rows have the documented regression targets; cards say 87 samples without row-selection explanation. |
| `PSI-10_Experimental table_Flame Design_Normal Flame.csv` | 207 | Not used by inverse-flame models. |
| `PSI-107_Experimental table_Spice.csv` | 469 | Seven raw Fuel strings; whitespace normalization merges the two 75%-propylene spellings into the card's six labels. The actual artifact embeds its six class labels. |
| `PSI-159_Experimental table_CFI-G_Inverse Flames.csv` | 1,015 | Different feature names/missingness from normal table; no model training source. |
| `PSI-159_Experimental table_CFI-G_Normal Flames.csv` | 179 | Different feature columns; model harmonization undocumented. |
| `PSI-69_Experimental table_FLEX.csv` | 274 | CSV uses initial O/N/CO; model card has O2/N2/CO2 inconsistency, while README and serialized preprocessor agree on O/N/CO. |

The archive contains no Python training scripts/notebooks. It does not provide authoritative NASA source URLs, retrieval metadata, or reproducible training code. Reported metrics in this table are **source-reported, not independently recomputed**, except that actual inference was reproduced for controlled example payloads.

## Integrated endpoints and examples

All prediction requests are POST JSON; all additional classifiers return a categorical label, not a fire/ignition probability. The sample predictions below are actual local artifact inference results. Every response includes status `AVAILABLE`, provenance status `UNVERIFIED`, validation metadata, limitations, and an experimental scientific warning.

### PSI-10 endpoints (existing)

Shared request:

```json
{
  "oxygen_mole_fraction": 0.37,
  "oxygen_mass_flow_mg_s": 5.11,
  "nitrogen_mass_flow_mg_s": 7.59,
  "total_mass_flow_mg_s": 12.71,
  "initial_pressure_atm": 0.99,
  "mean_pressure_atm": 1.0
}
```

| Method and path | Example output | Unit | Source-reported metric |
|---|---:|---|---|
| POST `/api/v1/predictions/psi10/adiabatic-flame-temperature` | `2531.907250860003` | K | MAE 53.50 K; R² 0.6218 |
| POST `/api/v1/predictions/psi10/stoichiometric-mixture-fraction` | `0.3281310072400002` | Inferred dimensionless fraction; card unit not specified | MAE 0.024039; R² 0.9305 |
| POST `/api/v1/predictions/psi10/pressure-rise` | `0.011819590199999999` | atm | MAE 0.001406 atm; R² 0.6030 |

All PSI-10 responses share `Psi10RegressionPredictionResponse`: `investigation_id`, `model_name`, `prediction_target`, `predicted_value`, `output_unit`, `evaluation_metrics`, `model_status`, `input_validation`, `provenance`, `limitations`, `scientific_warning`. The request uses six required finite numbers; mole fraction range 0–1, flow values >=0, pressures >0; extra fields are rejected.

### PSI-159 ignition (existing)

Request:

```json
{
  "flame_type": "Inverse",
  "fuel": "ethane",
  "pressure_bar": 1.1033,
  "adiabatic_flame_temperature_k": 2120.0,
  "hydrocarbon_burn_rate_mg_s": 1.5,
  "oxygen_flow_rate_mg_s": 0.85,
  "oxygen_mole_fraction": 0.15,
  "fuel_fraction": 0.2981
}
```

Response excerpt: `{"predicted_class":"Hot Ignition/Flash","predicted_class_code":1,"output_unit":null,"model_status":"AVAILABLE","provenance":{"verification_status":"UNVERIFIED"}}`. Reported metrics: test accuracy 90.48%; CV 88.88% ± 2.83%. No class probabilities are returned. The full request/response contract and feature order remain unchanged.

### PSI-159 burn time (existing)

Request:

```json
{
  "flame_type": "Normal",
  "fuel": "30% ethane",
  "pressure_bar": 0.506650,
  "adiabatic_flame_temperature_k": 2483.47,
  "hydrocarbon_burn_rate_mg_s": 0.75,
  "oxygen_flow_rate_mg_s": 0.85,
  "oxygen_mole_fraction": 0.15,
  "fuel_fraction": 0.300
}
```

Response: `predicted_burn_time_seconds: 196.84`. The existing burn-time contract and artifact are unchanged. Reported MAE 23.08 sec, RMSE 36.14 sec, R² 0.1575; evaluation sample/split details unavailable.

### PSI-107 fuel classifier (new)

Path: **POST `/api/v1/predictions/psi107/fuel-classification`**

```json
{
  "nozzle_mm": 1.6002,
  "fuel_rate_mg_s": 1.971,
  "hydrocarbon_fuel_rate_mg_s": 1.971,
  "air_rate_cm_s": 36.39
}
```

This documented request actually predicted `Propane`, class index `4`, according to the ordered class labels embedded in the model packet: `50%Propylene`, `75%Propylene`, `Ethylene`, `Methane`, `Propane`, `Propylene`. The response schema is `AdditionalClassificationResponse`, including class, metrics (source-reported accuracy 0.8191), status, validation, provenance, limitations, and scientific warning. All four inputs are required finite numbers. No source-supported domain ranges are documented; nozzle must be >0, fuel/HC rates >=0, and air rate permits signed values because the source contains negative values. No class probabilities are exposed.

### PSI-69 flame outcome (new)

Path: **POST `/api/v1/predictions/psi69/flame-outcome`**

```json
{
  "ambient_pressure_mmhg": 757.7,
  "oxygen_mole_fraction": 0.21,
  "nitrogen_mole_fraction": 0.79,
  "carbon_monoxide_mole_fraction": 0.0,
  "helium_mole_fraction": 0.0,
  "droplet_initial_diameter_mm": 2.1,
  "fuel": "Methanol"
}
```

Actual model inference returned `Extinction`, class index `2`, under the class order stated in its README: Completion, Disruption, Extinction. The response schema is `AdditionalClassificationResponse`; metrics are null because no numeric test score is documented. Pressure >=0 mmHg; mole fractions 0–1; diameter >0; fuel is Methanol or Heptane. Unknown fields are rejected. Softmax scores are intentionally not exposed as calibrated probabilities.

**Keras compatibility:** PSI-69 artifact metadata says Keras 3.13.2; compatible TensorFlow 2.16.2/Keras 3.10.0 loads it after removing only a null `quantization_config` key from the in-memory config and loading the unchanged weights. PSI-69 class order is from its inference README and is not supported by training code; the model card also conflicts on the O/N/CO feature naming. Responses disclose these limitations.

## Blocked model

**PSI-69 `01-Model` regression is not integrated.** Archive metadata shows an 8-input, 1-output linear MSE regressor and a paired preprocessor, but its README and model card are empty. Several dataset columns could be plausible continuous targets; no source ties this model to one. Target, output unit, metrics, and training split therefore cannot be established. No endpoint or copied artifacts were created for it.

## Runtime dependencies

The configured Python version is 3.9.6 on macOS arm64. Tested dependency versions: NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.6.1, joblib 1.5.3, PyTorch 2.6.0, TensorFlow 2.16.2, Keras 3.10.0. TensorFlow and PyTorch macOS-arm64 wheels were confirmed available for Python 3.9 before installation. `requirements.txt` pins torch, tensorflow and keras. PSI-107 and PSI-69 bundles are lazy-loaded, cached, SHA-256 checked, and validated for expected feature dimensions; failures return HTTP 503, inference errors HTTP 500.

## Verification and limitations

- Source archive: `Predictive-Models-main (1).zip`; its contents were inspected without running repository scripts.
- PSI-107 packet was deserialized after reviewing pickle globals and documentation; embedded class list/features/scaler were inspected. The documented example inference was reproduced.
- PSI-69/02 preprocessor and Keras config/weights were inspected. The README example was reproduced using its raw feature names and the documented class ordering. No original training code exists in the archive.
- PSI-69/01 is blocked as above.
- Existing five endpoint contracts remain registered. Model/source provenance is UNVERIFIED; none is NASA-verified or certified for spacecraft/lunar/Martian safety.
- Legacy compare/assistant/rank endpoints remain `NOT_IMPLEMENTED` stubs.

## Backend files and runtime validation

Newly packaged artifacts:

- `models/psi107/fuel_classifier_pipeline.joblib`
- `models/psi69/dnn_cls_model.keras`
- `models/psi69/dnn_cls_preprocessor.joblib`

New endpoints:

- `POST /api/v1/predictions/psi107/fuel-classification`
- `POST /api/v1/predictions/psi69/flame-outcome`

Implementation and regression tests are in `app/models/additional_predictions.py`, `app/services/additional_predictions.py`, `app/api/routes/predictions.py`, and `tests/test_remaining_predictions.py`. Existing five prediction endpoints were retained.

Final validation on the configured Python 3.9.6 environment: **108 tests passed**. `pip check` reported no broken requirements; `git diff --check` passed. Live TestClient smoke probes returned HTTP 200 for the seven prediction endpoints and OpenAPI/Swagger, with these candidate examples: PSI-107 `Propane`, PSI-69 `Extinction`. These pass results do not verify NASA provenance, model quality, or mission safety.

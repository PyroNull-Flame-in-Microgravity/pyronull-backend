# PyroNull Backend — FIREWISE AI

FastAPI backend for the PyroNull team's NASA Space Apps Challenge project, **Flame in Freefall**.

The product is an evidence-linked research dashboard to discover, compare and interpret NASA microgravity combustion investigations relevant to spacecraft fire prevention, detection and suppression. It is **not a certified spacecraft fire-risk predictor**.

## Current status
The API includes a fixture-backed Mission Intelligence module. The records in `data/raw/psi_full_dataset.csv` and `data/raw/psi_investigations_dataset.csv` are explicitly marked demo data and have **not** been verified against NASA PSI. Existing `/investigations`, `/compare`, `/assistant/query`, and `/rank` routes retain their initial demo/stub behavior.

## API
Base path: `/api/v1`

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/investigations` | List/filter investigations |
| GET | `/investigations/{id}` | Investigation detail |
| POST | `/compare` | Compare investigations/runs |
| POST | `/assistant/query` | Grounded research Q&A (not implemented yet) |
| POST | `/rank` | Research relevance ranking (not implemented yet) |
| GET | `/mission-intelligence/profiles` | List spacecraft, station, lunar, Mars, and rover mission profiles |
| POST | `/mission-intelligence/scenario` | Validate a mission scenario and retain unspecified conditions as unknown |
| POST | `/mission-intelligence/discover` | Discover fixture records with evidence and limitations |
| POST | `/mission-intelligence/rank` | Explainable heuristic research-relevance ordering |
| POST | `/mission-intelligence/gaps` | Report coverage gaps in the supplied CSV fixtures only |
| POST | `/mission-intelligence/report` | Combine a profile, scenario, ranking, evidence, limitations, and gaps |
| POST | `/predictions/burn-time` | Experimental PSI-159 model burn-time estimate |
| POST | `/predictions/psi10/adiabatic-flame-temperature` | PSI-10 inverse-flame Tad estimate |
| POST | `/predictions/psi10/stoichiometric-mixture-fraction` | PSI-10 inverse-flame mean Zst estimate |
| POST | `/predictions/psi10/pressure-rise` | PSI-10 inverse-flame pressure-rise estimate |
| POST | `/predictions/psi159/ignition` | PSI-159 ignition class (not an ignition/fire probability) |
| POST | `/predictions/psi107/fuel-classification` | PSI-107 experimental fuel class |
| POST | `/predictions/psi69/flame-outcome` | PSI-69 FLEX experimental flame outcome class |

The legacy `/compare`, `/assistant/query`, and `/rank` routes currently return HTTP 200 with `status: "NOT_IMPLEMENTED"` and empty/stub result fields. Frontend clients must check `status` and must not interpret HTTP 200 as completed analysis. Existing request and response shapes are retained.

### PSI-159 burn-time prediction

`POST /api/v1/predictions/burn-time` accepts JSON using these API field names: `flame_type`, `fuel`, `pressure_bar`, `adiabatic_flame_temperature_k`, `hydrocarbon_burn_rate_mg_s`, `oxygen_flow_rate_mg_s`, `oxygen_mole_fraction`, and `fuel_fraction`. Pressure is in bar, temperature in kelvin, flow/burn rates in mg/s, and mole fractions are from 0 to 1.

Example request:

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

Example response shape:

```json
{
	"predicted_burn_time_seconds": 196.84,
	"model_name": "PSI-159 Burn Time Regressor",
	"model_type": "Scikit-learn Random Forest regression pipeline",
	"evaluation_metrics": {"mae_seconds": 23.08, "rmse_seconds": 36.14, "r_squared": 0.1575},
	"model_limitations": ["Reported test-set errors are substantial; individual prediction errors may be larger."],
	"experimental_domain_warning": "Experimental estimate only; not a fire-risk prediction or habitat safety assessment."
}
```

The model artifact is loaded lazily once per application process. The prediction is an experimental burn-time estimate—not a fire-risk prediction, fire probability, suppression-time prediction, or habitat safety assessment. It is not validated for lunar, Martian, spacecraft, or other environments outside its training domain. Reported MAE (23.08 seconds), RMSE (36.14 seconds), and R² (0.1575) come from the accompanying model documentation and have not been independently reproduced by this API.

## Additional research prediction models

The following seven selected models are available in the prediction API: three PSI-10 models, two PSI-159 models, PSI-107 fuel classification, and PSI-69/02 flame outcome. Every successful response includes `model_status: "AVAILABLE"` and `provenance.verification_status: "UNVERIFIED"`. Model-card metrics are documentation-reported and were not independently reproduced unless explicitly stated. These are experimental research estimates/classifications, not certified fire-safety outputs. The new artifacts were copied from the locally extracted Predictive-Models repository after inspecting their documentation and serialization metadata; their upstream source and NASA provenance have not been independently verified. Artifact SHA-256 values are checked before lazy per-process loading.

### Inventory of candidate models

| Investigation/model | Status | Backend artifact | Dataset referenced by supplied docs | Target and output | Documented evaluation | Verification / decision |
|---|---|---|---|---|---|---|
| PSI-10 inverse-flame adiabatic temperature | Integrated | `models/psi10/inverse_flame_tad_predictor.joblib` | `PSI-10_Experimental table_Flame Design_Inverse Flame.csv` (120 rows in extracted repo) | `Tad`; K | MAE 53.50 K; R² 0.6218; card says 87 samples, 20% test split; exact partitions unavailable | UNVERIFIED; card/CSV sample counts disagree and data/source URL are unconfirmed |
| PSI-10 inverse-flame stoichiometric mixture fraction | Integrated | `models/psi10/inverse_flame_zst_predictor.joblib` | Same PSI-10 inverse-flame CSV | `mean Zst`; dimensionless is inferred from target name (unit not explicit in card) | MAE 0.024039; R² 0.9305; card says 87 samples, 20% test split; exact partitions unavailable | UNVERIFIED; card/CSV sample counts disagree and data/source URL are unconfirmed |
| PSI-10 inverse-flame pressure rise | Integrated | `models/psi10/inverse_flame_pressure_rise_predictor.joblib` | Same PSI-10 inverse-flame CSV | `pfinal - pinit`; atm | MAE 0.001406 atm; R² 0.6030; card says 87 samples, 20% test split; exact partitions unavailable | UNVERIFIED; card/CSV sample counts disagree and data/source URL are unconfirmed; not a containment limit |
| PSI-159 ignition classification | Integrated | `models/psi159/PSI_159_ignition_classifier_Predictive_Model.joblib` | `PSI-159_Experimental table_CFI-G_Inverse Flames.csv` and `PSI-159_Experimental table_CFI-G_Normal Flames.csv` | No ignition vs. hot ignition/flash; categorical class | Test accuracy 90.48%; five-fold CV 88.88% ± 2.83%; sample and split counts not given | UNVERIFIED; class output is not a real-world fire probability; source tables have differing schemas |
| PSI-159 burn-time regression | Already integrated at `/api/v1/predictions/burn-time` | `models/PSI_159_burntime_regressor_Predictive_Model.joblib` | Same two PSI-159 CFI-G CSVs | Burn time; seconds | MAE 23.08 s; RMSE 36.14 s; R² 0.1575; sample/split counts not given | UNVERIFIED; see burn-time limitations above |
| PSI-107 fuel classification | Integrated | `models/psi107/fuel_classifier_pipeline.joblib` | `PSI-107_Experimental table_Spice.csv` (469 rows in extracted repo) | Fuel class | Card reports accuracy 0.8191; split/sample counts not documented | UNVERIFIED; class sequence read from artifact and actual inference tested; raw CSV has whitespace label variants |
| PSI-69 flame outcome classification (`02-Model`) | Integrated | `models/psi69/dnn_cls_model.keras` + `models/psi69/dnn_cls_preprocessor.joblib` | `PSI-69_Experimental table_FLEX.csv` (274 rows in extracted repo) | `Test end`: Completion, Disruption, Extinction | No numerical test metric documented; card states 80/20 split and training oversampling to 447 rows, exact partition sizes absent | UNVERIFIED; actual inference tested; model-card O2/N2/CO2 names conflict with actual preprocessor O/N/CO names; class order from README only |
| PSI-69 `01-Model` artifact (possible regression candidate) | BLOCKED | `dnn_model.keras` + `dnn_preprocessor.joblib` not copied | No usable provenance/model card | Target, output unit, and feature schema unknown | Not documented | BLOCKED: accompanying README and model card are empty, so model purpose/schema cannot be established |

The extracted PSI-10 repository included 120 rows in the named inverse-flame CSV while each model card reports 87 samples and a 20% test partition without explaining row selection or exact train/test counts. The PSI-159 docs do not report sample counts. No downloaded-repository CSVs were copied into this backend. `VERIFIED` means original source/data independently confirmed; none of these downloaded-repository models currently meets that standard.

#### Extracted dataset inventory (not copied into the backend)

| Dataset filename | Rows | Relevance / noted metadata issue |
|---|---:|---|
| `PSI-10_Experimental table_Flame Design_Inverse Flame.csv` | 120 | Model cards say 87 samples; row selection is unexplained |
| `PSI-10_Experimental table_Flame Design_Normal Flame.csv` | 207 | Not used by the integrated inverse-flame regressors |
| `PSI-107_Experimental table_Spice.csv` | 469 | Fuel labels include whitespace variants; artifact embeds a six-label sequence |
| `PSI-159_Experimental table_CFI-G_Inverse Flames.csv` | 1,015 | Several feature/target fields have substantial missingness; no training-row count is documented |
| `PSI-159_Experimental table_CFI-G_Normal Flames.csv` | 179 | Feature columns differ from the inverse-flame table |
| `PSI-69_Experimental table_FLEX.csv` | 274 | Card describes an 80/20 split and SMOTE-balanced training set of 447; no numerical test metrics are supplied |

The extracted repository supplied no independently confirmed NASA source URLs or retrieval metadata for these model/dataset bundles. Folder names and README statements alone do not establish NASA verification.

Tested runtime additions are PyTorch 2.6.0, TensorFlow 2.16.2, and Keras 3.10.0 on Python 3.9 macOS arm64. PSI-69's artifact declares Keras 3.13.2; its nullable `quantization_config` serialization field is removed only in memory to load the same architecture and unchanged weights with this runtime. `requirements.txt` pins the tested framework versions.

All three PSI-10 endpoints use the documented input schema below. Values must be finite; fractions are constrained to [0, 1], flows to nonnegative mg/s, and pressures to positive atm. No training-distribution check is claimed.

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

#### `POST /api/v1/predictions/psi10/adiabatic-flame-temperature`

Example response excerpt (the actual response also includes input validation and complete limitations/provenance notes):

```json
{
	"investigation_id": "PSI-10",
	"model_name": "PSI-10 Inverse Flame Adiabatic Temperature Predictor",
	"prediction_target": "Adiabatic flame temperature (Tad)",
	"predicted_value": 2531.90725086,
	"output_unit": "K",
	"evaluation_metrics": {"mae": 53.5, "mae_unit": "K", "r_squared": 0.6218, "documented_sample_count": 87, "documented_test_split_ratio": 0.2},
	"model_status": "AVAILABLE",
	"provenance": {"dataset_filename": "PSI-10_Experimental table_Flame Design_Inverse Flame.csv", "source_url": null, "verification_status": "UNVERIFIED"},
	"scientific_warning": "Experimental research-model output only; not a certified spacecraft fire-safety assessment."
}
```

#### `POST /api/v1/predictions/psi10/stoichiometric-mixture-fraction`

Request: the shared PSI-10 JSON above. Example response excerpt:

```json
{
	"investigation_id": "PSI-10",
	"model_name": "PSI-10 Inverse Flame Stoichiometric Mixture Fraction Predictor",
	"prediction_target": "Mean stoichiometric mixture fraction (mean Zst)",
	"predicted_value": 0.32813100724,
	"output_unit": "dimensionless fraction (unit implied by target name)",
	"evaluation_metrics": {"mae": 0.024039, "mae_unit": "dimensionless fraction", "r_squared": 0.9305, "documented_sample_count": 87, "documented_test_split_ratio": 0.2},
	"model_status": "AVAILABLE",
	"provenance": {"verification_status": "UNVERIFIED"}
}
```

#### `POST /api/v1/predictions/psi10/pressure-rise`

Request: the shared PSI-10 JSON above. Example response excerpt:

```json
{
	"investigation_id": "PSI-10",
	"model_name": "PSI-10 Inverse Flame Pressure Rise Predictor",
	"prediction_target": "Pressure rise (pfinal - pinit)",
	"predicted_value": 0.0118195902,
	"output_unit": "atm",
	"evaluation_metrics": {"mae": 0.001406, "mae_unit": "atm", "r_squared": 0.603, "documented_sample_count": 87, "documented_test_split_ratio": 0.2},
	"model_status": "AVAILABLE",
	"provenance": {"verification_status": "UNVERIFIED"}
}
```

Do not use this as a pressure containment limit.

### `POST /api/v1/predictions/psi159/ignition`

Required inputs: `flame_type` and `fuel` categorical strings; `pressure_bar` in bar; `adiabatic_flame_temperature_k` in K; `hydrocarbon_burn_rate_mg_s` and `oxygen_flow_rate_mg_s` in mg/s; `oxygen_mole_fraction` (`XO2`) and `fuel_fraction` (`Xf`) in [0, 1].

Example request:

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

Example response excerpt:

```json
{
	"investigation_id": "PSI-159",
	"model_name": "PSI-159 Ignition / Flammability Classifier",
	"prediction_target": "Hot ignition / flash classification",
	"predicted_class": "Hot Ignition/Flash",
	"predicted_class_code": 1,
	"output_unit": null,
	"evaluation_metrics": {"accuracy": 0.9048, "cross_validation_accuracy": 0.8888, "cross_validation_standard_deviation": 0.0283},
	"model_status": "AVAILABLE",
	"provenance": {"verification_status": "UNVERIFIED"}
}
```

The API does not expose class probabilities; class labels must not be interpreted as real-world fire probabilities.

### `POST /api/v1/predictions/psi107/fuel-classification`

The model's joblib packet embeds its class order; the API uses that order rather than inferring it from CSV labels. Required finite numeric inputs are `nozzle_mm` (>0, mm), `fuel_rate_mg_s` (>=0, mg/s), `hydrocarbon_fuel_rate_mg_s` (>=0, mg/s), and `air_rate_cm_s` (cm/s; signed source values are permitted). The documented sample predicts class `Propane`, index 4. The model card reports accuracy 0.8191, with no sample count or split details.

Example request:

```json
{
	"nozzle_mm": 1.6002,
	"fuel_rate_mg_s": 1.971,
	"hydrocarbon_fuel_rate_mg_s": 1.971,
	"air_rate_cm_s": 36.39
}
```

Example response excerpt:

```json
{
	"investigation_id": "PSI-107",
	"model_name": "PSI-107 FuelClassifierDNN",
	"prediction_target": "Fuel class",
	"predicted_class": "Propane",
	"predicted_class_index": 4,
	"output_unit": null,
	"evaluation_metrics": {"accuracy": 0.8191},
	"model_status": "AVAILABLE",
	"provenance": {"dataset_filename": "PSI-107_Experimental table_Spice.csv", "verification_status": "UNVERIFIED"}
}
```

The API deliberately does not return logits/probabilities as calibrated class confidence. A class label is not a real-world fuel-detection or fire probability.

### `POST /api/v1/predictions/psi69/flame-outcome`

Inputs: `ambient_pressure_mmhg` (finite >=0, mmHg), `oxygen_mole_fraction`, `nitrogen_mole_fraction`, `carbon_monoxide_mole_fraction`, and `helium_mole_fraction` (finite fractions in [0,1]), `droplet_initial_diameter_mm` (finite >0, mm), and `fuel` (`Methanol` or `Heptane`). The API follows the actual serialized preprocessor's raw feature names O/N/CO. The model card's O2/N2/CO2 field names conflict with those source/preprocessor names. The model expects eight encoded inputs after preprocessing. The README documents class order `Completion`, `Disruption`, `Extinction`; no training script was found to verify this order. The model card reports 274 rows and an 80/20 split with SMOTE on training data, but no numeric evaluation score.

Example request:

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

This example was tested with the actual model and returned `Extinction` (class index 2). The response includes `investigation_id: "PSI-69"`, model name, `prediction_target: "Test end outcome"`, `predicted_class`, `predicted_class_index`, `output_unit: null`, metrics (accuracy fields null because none are supplied), `model_status`, input validation, provenance, limitations, and scientific warning. Softmax scores are intentionally not exposed as calibrated outcome or fire probabilities.

The Python 3.9 macOS-arm64 runtime was tested with pinned PyTorch 2.6.0, TensorFlow 2.16.2, and Keras 3.10.0. The PSI-69 artifact declares Keras 3.13.2; its nullable `quantization_config` key is removed in-memory before reconstructing the unchanged architecture and loading unchanged weights. Both new model bundles are lazy-loaded, cached, SHA-256 checked, and return controlled 503 errors when artifacts or compatibility checks fail.

All prediction responses include `input_validation`, `limitations`, and `scientific_warning`. Try all available endpoints in Swagger at `/docs`. All model/data provenance remains UNVERIFIED pending independent source verification.

Health: `/health` · Swagger: `/docs` · ReDoc: `/redoc`

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```
Open `http://localhost:8000/docs`.

### Frontend CORS configuration

The development default allows `http://localhost:3000` and `http://localhost:5173`. For deployment, set `CORS_ALLOWED_ORIGINS` to a comma-separated list of exact frontend origins, including scheme and optional port, for example `https://app.example.org`. With a `.env` file, start Uvicorn using `--env-file .env`; otherwise set the variable in the deployment environment before starting the process. Do not include URL paths or trailing slashes.

## NASA data priorities
The team guide recommends starting with BASS-II (prevention), SAME-R (detection), and FLEX (suppression), with Saffire as supporting context. Verify exact records and downloadable files on NASA's live PSI Combustion Science repository. Never guess record IDs or direct file URLs.

## Scientific data integrity
- Keep raw source files unchanged and processed outputs separate.
- Record source URL, exact investigation identifier, file name/type and retrieval date.
- Preserve source units and document conversions.
- Missing values are not zero; label them "not reported" or "not available in source".
- Distinguish NASA measurements, reported findings, derived values and AI-generated explanations.
- Cite sources in AI answers and dashboard facts.
- Do not claim lunar/Martian partial-gravity experimental validation without supporting data.
- Research relevance is not fire probability, a hazard score or safety certification.
- Mission Intelligence relevance values are unvalidated heuristics, not fire-risk or safety scores. Unknown fixture fields are not treated as zero or as proof that research does not exist.


Never commit credentials, `.env`, private data or large raw NASA files. Commit `.env.example` instead.
# pyronull-backend
Backend API, NASA data integration, scientific data processing, and analytical services for PyroNull — a fire safety and combustion science platform.

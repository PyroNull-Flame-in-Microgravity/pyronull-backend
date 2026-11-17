# PyroNull Predictive Lab — Backend API Handoff

**Audience:** frontend developer integrating the Predictive Lab. This handoff describes the current backend contracts; it does not implement frontend code. The backend workspace does not contain `PredictiveLabPage.tsx` or a frontend project, so frontend file paths below are recommendations to adapt to the teammate's repository.

## Backend verification summary

- Prediction API base path: `/api/v1/predictions`.
- Seven POST prediction endpoints are registered and tested: PSI-159 burn time, three PSI-10 regressors, PSI-159 ignition classification, PSI-107 fuel classification, and PSI-69 flame outcome.
- Swagger: `http://localhost:8001/docs`; OpenAPI: `http://localhost:8001/openapi.json`.
- Actual inference for all seven endpoints passed in the previously verified local container. The current 133-test suite also exercises local model loading, including PSI-107/PSI-69 under their pinned SHA-256 checks. Those hashes identify the current bytes but do not establish source provenance or redistribution rights.
- The three PSI-107/PSI-69 artifacts are excluded from the Git and Docker contexts pending written rights clearance. Their local inference results do not make a standard public Git checkout or full seven-model image reproducible; the Docker artifact guard will fail with their exact missing paths.
- All model provenance statuses are `UNVERIFIED`. The local model repository and cards do not independently establish NASA source verification or scientific validation.
- Successful predictions are experimental outputs, not certified spacecraft fire-safety assessments. Do not frame classifier labels as real-world ignition/fire probabilities.
- The current suite passed 133 tests with two warnings after the burn-time integrity and Docker allowlist changes. The Gemini provider tests use a mocked SDK client; no live Gemini request was made during this verification.

## Base URL, CORS, and request handling

Set a frontend environment variable such as:

```text
VITE_API_BASE_URL=http://localhost:8001
```

Have the existing frontend API client build requests as `${VITE_API_BASE_URL}/api/v1/...`. `VITE_API_BASE_URL` should be the origin only, without `/api/v1`, unless the client deliberately treats it as a complete API base. Do not hardcode the host in each page/card.

Development CORS defaults permit `http://localhost:3000` and `http://localhost:5173`. For a hosted frontend, configure the backend process with `CORS_ALLOWED_ORIGINS` as a comma-separated list of exact origins, including scheme and port, e.g. `https://lab.example.org`. Restart the backend after changing it. The app reads process environment variables at import/startup; for a `.env` file, launch Uvicorn with `--env-file .env` (or configure the variable in the deployment platform).

Standard client behavior:

- **200:** prediction response; show the experimental/provenance warnings returned in the payload.
- **422:** invalid, missing, non-finite, out-of-range, or extra request fields; show field validation errors and let the user correct input.
- **503:** requested model artifact is missing, fails its pinned SHA-256 check, or cannot load due to incompatibility; show model unavailable and do not silently substitute another model.
- **500:** the model failed inference or returned an unusable result; show a generic retry/error state and retain diagnostics only in developer tooling.
- Network/CORS failure: distinguish from a model response; show backend unavailable and check configured API origin/CORS.

FastAPI/Pydantic errors normally use `{ "detail": [...] }` for 422 and `{ "detail": "..." }` for handled 500/503 responses. Do not assume every validation detail has the same message text; key off status and `loc` when present.

## Shared UI guidance

Each prediction should have its own card/form. Keep server prediction output distinct from user-provided inputs. Show `model_name`, `model_status`, evaluation metrics, `provenance.verification_status`, limitations, and scientific warning together with the result. Add a visible **UNVERIFIED / experimental** badge. The models have no API-provided prediction interval; evaluation MAE is a dataset-level metric, not an individual confidence interval. Do not convert an experimental predicted value into risk, probability, safety, suppression, or certification language.

All endpoints are POST; their full JSON Schemas are in Swagger. The additional PSI-10 and ignition routes reject unknown fields (`extra: forbid`) and reject NaN/Infinity. The PSI-159 burn-time schema also rejects extras and non-finite values. Numeric JSON fields should be sent as JSON numbers, not localized strings.

---

## 1. PSI-159 Burn Time Prediction

**Frontend card:** “Burn Time Estimate” / PSI-159. Existing backend implementation; contract preserved.

**Endpoint:** `POST /api/v1/predictions/burn-time`

**Inputs:**

| API field | Type / constraints | Unit / meaning |
|---|---|---|
| `flame_type` | string, 1–100 characters, nonblank after trimming | Experimental flame label, e.g. `Normal` |
| `fuel` | string, 1–120 characters, nonblank after trimming | Fuel label, e.g. `30% ethane` |
| `pressure_bar` | finite number, `> 0` | bar |
| `adiabatic_flame_temperature_k` | finite number, `> 0` | K |
| `hydrocarbon_burn_rate_mg_s` | finite number, `>= 0` | mg/s |
| `oxygen_flow_rate_mg_s` | finite number, `>= 0` | mg/s |
| `oxygen_mole_fraction` | finite number, 0–1 inclusive | `XO2` mole fraction |
| `fuel_fraction` | finite number, 0–1 inclusive | `Xf` fuel fraction |

**Example request:**

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

**Example response:**

```json
{
  "predicted_burn_time_seconds": 196.84,
  "model_name": "PSI-159 Burn Time Regressor",
  "model_type": "Scikit-learn Random Forest regression pipeline",
  "evaluation_metrics": {
    "mae_seconds": 23.08,
    "rmse_seconds": 36.14,
    "r_squared": 0.1575
  },
  "model_limitations": [
    "Documented test-set MAE is approximately 23.08 seconds and RMSE is approximately 36.14 seconds; individual prediction errors may be larger.",
    "Documented R-squared is approximately 0.1575, indicating limited explanatory performance on the reported evaluation set.",
    "Evaluation metrics are reported from the accompanying model documentation and have not been independently reproduced by this API.",
    "Predictions depend on the serialized model artifact and its training data; no uncertainty interval is produced."
  ],
  "experimental_domain_warning": "This is an experimental burn-time estimate for inputs resembling the PSI-159 model's training domain. It is not a fire-risk prediction, fire probability, suppression-time prediction, or habitat safety assessment. It is not validated for lunar, Martian, spacecraft, or other environments outside the training domain."
}
```

**Display:** show seconds, the documented MAE/RMSE/R², and a prominent warning. In particular, R² ≈ 0.1575 and MAE ≈ 23.08 seconds indicate substantial limitations. Not a fire-risk, fire-probability, suppression-time, or habitat-safety prediction. The model's reported evaluation metrics and NASA/source provenance are unverified by this API.

**Backend validation/errors:** missing fields, wrong types, blank categories, invalid ranges, NaN/Infinity, or extra fields yield 422. Missing or unloadable artifact yields 503; inference failure yields 500.

---

## 2–4. PSI-10 Inverse-Flame Regressors

All three use the same request schema but different target/artifact. These are separate cards/routes; do not combine their outputs into one metric. Inputs are required finite JSON numbers; `oxygen_mole_fraction` is in [0,1], flow values are nonnegative mg/s, and pressures are positive atm. Extra fields are rejected. The backend does not validate membership in the model training distribution.

**Shared PSI-10 request example** (send this body to each of the three routes):

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

The model feature order is `XO2`, `mO2 (mg/s)`, `mN2 (mg/s)`, `mtot (mg/s)`, `pinit (atm)`, `pmean (atm)`.

### PSI-10 Adiabatic Flame Temperature

- **Purpose/card:** estimate inverse-flame adiabatic flame temperature (`Tad`).
- **Endpoint:** `POST /api/v1/predictions/psi10/adiabatic-flame-temperature`.
- **Output/display:** `predicted_value` as K, named by `prediction_target`; show MAE 53.50 K and R² 0.6218 as reported model-card metrics.
- **Artifact:** `models/psi10/inverse_flame_tad_predictor.joblib`.
- **Example response excerpt:**

```json
{
  "investigation_id": "PSI-10",
  "model_name": "PSI-10 Inverse Flame Adiabatic Temperature Predictor",
  "prediction_target": "Adiabatic flame temperature (Tad)",
  "predicted_value": 2531.907250860003,
  "output_unit": "K",
  "evaluation_metrics": {
    "mae": 53.5,
    "mae_unit": "K",
    "rmse": null,
    "rmse_unit": null,
    "r_squared": 0.6218,
    "accuracy": null,
    "cross_validation_accuracy": null,
    "cross_validation_standard_deviation": null,
    "documented_sample_count": 87,
    "documented_test_split_ratio": 0.2,
    "sample_count_note": "Model card reports 87 samples and a 20% test split but does not state exact partition counts; the associated inverse-flame CSV has 120 rows and the difference is unexplained."
  },
  "model_status": "AVAILABLE",
  "provenance": {
    "dataset_filename": "PSI-10_Experimental table_Flame Design_Inverse Flame.csv",
    "source_url": null,
    "verification_status": "UNVERIFIED"
  }
}
```

### PSI-10 Stoichiometric Mixture Fraction

- **Purpose/card:** estimate mean `Zst` for inverse flames.
- **Endpoint:** `POST /api/v1/predictions/psi10/stoichiometric-mixture-fraction`.
- **Output/display:** `predicted_value`; the card does not explicitly state a unit. The response describes it as a dimensionless fraction inferred from the target name. Show MAE 0.024039 and R² 0.9305.
- **Artifact:** `models/psi10/inverse_flame_zst_predictor.joblib`.
- **Example response excerpt:**

```json
{
  "investigation_id": "PSI-10",
  "model_name": "PSI-10 Inverse Flame Stoichiometric Mixture Fraction Predictor",
  "prediction_target": "Mean stoichiometric mixture fraction (mean Zst)",
  "predicted_value": 0.3281310072400002,
  "output_unit": "dimensionless fraction (unit implied by target name)",
  "evaluation_metrics": {
    "mae": 0.024039,
    "mae_unit": "dimensionless fraction",
    "rmse": null,
    "rmse_unit": null,
    "r_squared": 0.9305,
    "accuracy": null,
    "cross_validation_accuracy": null,
    "cross_validation_standard_deviation": null,
    "documented_sample_count": 87,
    "documented_test_split_ratio": 0.2,
    "sample_count_note": "Model card reports 87 samples and a 20% test split but does not state exact partition counts; the associated inverse-flame CSV has 120 rows and the difference is unexplained."
  },
  "model_status": "AVAILABLE",
  "provenance": {"verification_status": "UNVERIFIED"}
}
```

### PSI-10 Pressure Rise

- **Purpose/card:** estimate `pfinal - pinit` for inverse flames.
- **Endpoint:** `POST /api/v1/predictions/psi10/pressure-rise`.
- **Output/display:** `predicted_value` in atm; show MAE 0.001406 atm and R² 0.6030. **Do not present as a containment limit or safety threshold.**
- **Artifact:** `models/psi10/inverse_flame_pressure_rise_predictor.joblib`.
- **Example response excerpt:**

```json
{
  "investigation_id": "PSI-10",
  "model_name": "PSI-10 Inverse Flame Pressure Rise Predictor",
  "prediction_target": "Pressure rise (pfinal - pinit)",
  "predicted_value": 0.011819590199999999,
  "output_unit": "atm",
  "evaluation_metrics": {
    "mae": 0.001406,
    "mae_unit": "atm",
    "rmse": null,
    "rmse_unit": null,
    "r_squared": 0.603,
    "accuracy": null,
    "cross_validation_accuracy": null,
    "cross_validation_standard_deviation": null,
    "documented_sample_count": 87,
    "documented_test_split_ratio": 0.2,
    "sample_count_note": "Model card reports 87 samples and a 20% test split but does not state exact partition counts; the associated inverse-flame CSV has 120 rows and the difference is unexplained."
  },
  "model_status": "AVAILABLE",
  "provenance": {"verification_status": "UNVERIFIED"}
}
```

All PSI-10 responses also include `input_validation`, `provenance.note`, `limitations`, and `scientific_warning`; excerpts above omit unchanged common fields for readability.

**Validation/errors for all PSI-10 cards:** a missing required field, non-number, NaN/Infinity, invalid fraction/flow/pressure, or extra field returns 422. Missing/changed/incompatible artifact returns 503; inference error returns 500.

---

## 5. PSI-159 Ignition Classification

- **Purpose/card:** classify the experimental outcome as no ignition or hot ignition/flash.
- **Endpoint:** `POST /api/v1/predictions/psi159/ignition`.
- **Artifact:** `models/psi159/PSI_159_ignition_classifier_Predictive_Model.joblib`.
- **Inputs / model order:**
  1. `flame_type`: nonblank string, max 80 chars (example `Inverse`).
  2. `fuel`: nonblank string, max 120 chars (example `ethane`).
  3. `pressure_bar`: finite number `>0`, bar.
  4. `adiabatic_flame_temperature_k`: finite number `>0`, K.
  5. `hydrocarbon_burn_rate_mg_s`: finite number `>=0`, mg/s.
  6. `oxygen_flow_rate_mg_s`: finite number `>=0`, mg/s.
  7. `oxygen_mole_fraction`: finite number in [0,1], `XO2`.
  8. `fuel_fraction`: finite number in [0,1], `Xf`.
- Unknown request fields are rejected. Numeric inputs outside these basic constraints return 422; training-domain membership is not verified.
- **Response/display:** `predicted_class` is `No Ignition` or `Hot Ignition/Flash`; `predicted_class_code` is 0 or 1. `output_unit` is null. Documented metrics are test accuracy 0.9048 and cross-validation accuracy 0.8888 ± 0.0283; sample/split counts are unavailable. The endpoint intentionally does not return class probabilities, even though the estimator has a probability method, to avoid implying calibrated real-world ignition/fire probabilities.

**Example request:**

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

**Example response excerpt:**

```json
{
  "investigation_id": "PSI-159",
  "model_name": "PSI-159 Ignition / Flammability Classifier",
  "prediction_target": "Hot ignition / flash classification",
  "predicted_class": "Hot Ignition/Flash",
  "predicted_class_code": 1,
  "output_unit": null,
  "evaluation_metrics": {
    "accuracy": 0.9048,
    "cross_validation_accuracy": 0.8888,
    "cross_validation_standard_deviation": 0.0283,
    "sample_count_note": "The accompanying README reports test accuracy and five-fold cross-validation but gives no dataset, training, or test sample counts."
  },
  "model_status": "AVAILABLE",
  "provenance": {
    "dataset_filename": "PSI-159_Experimental table_CFI-G_Inverse Flames.csv and PSI-159_Experimental table_CFI-G_Normal Flames.csv",
    "source_url": null,
    "verification_status": "UNVERIFIED"
  }
}
```

The complete response also includes `input_validation`, limitations, and a warning that this classification is not a fire probability.

---

## Loading, model-list endpoint, and compatibility

- `GET /api/v1/predictions/models` **does not exist** (confirmed as 404). It is not required for a static frontend card mapping; the backend currently exposes seven prediction endpoints. A future dynamic model catalog would be a separate backend feature and should first define whether it lists only statically registered models or probes artifact readiness.
- All seven endpoint loaders (including the PSI-159 burn-time loader) check their pinned SHA-256 before deserialization and cache successful loads per process. A missing, altered, or incompatible model returns 503.
- `/docs` and `/openapi.json` are available. Request and response schemas are attached to the seven prediction POST routes.
- `/compare` and `/rank` remain `NOT_IMPLEMENTED` stubs. `/assistant/query` now uses Gemini with local fixture context; when matching context exists but Gemini is not configured, it returns HTTP 503. Fixture sources are labeled unverified, and the assistant is not a verified NASA document index. Mission Intelligence ranking behavior remains unchanged.
- All model/source provenance is `UNVERIFIED`. PSI-10 cards report 87 samples with 20% test split while the cited inverse CSV contains 120 rows and does not explain the training subset. PSI-159 model documentation provides no sample/split counts. Do not imply NASA verification or scientific validation for mission environments.

## Frontend integration tasks (frontend-owned)

The backend workspace contains no `PredictiveLabPage.tsx`, frontend API client, or React project, so these are integration targets to locate in the frontend repository rather than verified current paths:

- Map frontend cards to the required prediction endpoints, which currently number seven; do not merge output/units across models. PSI-107 and PSI-69 details are in `PREDICTIVE_MODELS_INTEGRATION_REPORT.md`.
- Update the existing API client/service module (for example `src/services/api.ts` or `src/lib/api.ts`) to centralize `${VITE_API_BASE_URL}/api/v1` requests and JSON/error handling.
- Add TypeScript request/response interfaces in the frontend's existing API types location; model PSI-10 shared inputs separately from PSI-159 categorical/numeric fields.
- Add UI tests for request mapping, validation display, loading/success/422/500/503/network states, and scientific warning visibility.
- Add the frontend deployment origin to the backend `CORS_ALLOWED_ORIGINS` environment configuration.

Suggested card behavior:

| Card | Display recommendation | Special warning |
|---|---|---|
| PSI-159 Burn Time | Estimated seconds, plus MAE/RMSE/R²; say errors may be large and no uncertainty interval is returned | Not risk, probability, suppression time, or habitat safety; low documented R² |
| PSI-10 Tad | K, MAE, R²; inverse-flame model only | Card/CSV sample-count mismatch; experimental-domain limitation |
| PSI-10 mean Zst | Numeric fraction, label unit as inferred/undocumented, MAE and R² | Unit is not explicit in supplied model card |
| PSI-10 Pressure Rise | atm, MAE and R² | Never show as a pressure safety/containment limit |
| PSI-159 Ignition | Categorical class and code, accuracy metrics | Not probability; no class-probability field is returned |

## Ownership checklist

### Backend completed

- [x] Seven prediction POST routes have explicit request/response schemas and stable paths.
- [x] Actual local inference was exercised for all seven artifacts in the prior local container verification. The current public Git/Docker context intentionally excludes three artifacts, so all-seven deployment is not reproducible until rights and delivery are cleared.
- [x] Request field mapping, units, validation and controlled HTTP errors are covered by tests.
- [x] Swagger/OpenAPI, default development CORS, existing prediction routes, Mission Intelligence routes, and legacy route paths were checked.
- [x] No frontend files were modified.

### Frontend developer remaining

- [ ] Locate/edit the actual `PredictiveLabPage.tsx` and existing API client/types in the frontend repository.
- [ ] Connect to the configured backend origin and preserve `/api/v1` path exactly once.
- [ ] Implement per-card forms using these request bodies and format units/metrics.
- [ ] Show UNVERIFIED/experimental warnings and do not reinterpret classes or numeric outputs as safety scores/probabilities.
- [ ] Handle 422, 500, 503, network/CORS and loading states.
- [ ] Configure production frontend origin in backend `CORS_ALLOWED_ORIGINS`.
- [ ] Test the frontend against Swagger/dev backend and a deployment-like origin.

### Remaining blockers / caveats

- [ ] Independent NASA/source/dataset verification and model scientific validation remain outstanding; no model is marked verified.
- [ ] `/api/v1/predictions/models` is absent, but is not a blocker for a static frontend card mapping.
- [ ] Frontend source was not present in the backend workspace, so exact frontend file paths/components beyond the named `PredictiveLabPage.tsx` cannot be confirmed here.
- [ ] Compare/rank remain explicit stubs and are outside the Predictive Lab integration. The assistant is Gemini-backed but does not have a verified NASA document corpus.

**Handoff status:** backend contracts are ready for frontend integration. This is not a claim of production readiness or NASA/scientific verification.

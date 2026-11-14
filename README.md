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

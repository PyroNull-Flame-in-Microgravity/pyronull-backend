# Docker, Gemini, and Render deployment

## Architecture and model inventory

The service is FastAPI (`app.main:app`) with routes mounted under `/api/v1` and a liveness endpoint at `/health`. Model inference is lazy and cached per process. The container preserves the existing `app/`, `models/`, and `data/raw/` paths. It packages only runtime CSV fixtures, not source research tables, scripts, tests, or documentation.

| Model | Endpoint | Artifact | Required runtime | Local inference verification | Deployment readiness |
| --- | --- | --- | --- | --- | --- |
| PSI-159 burn-time regressor | `POST /api/v1/predictions/burn-time` | `models/PSI_159_burntime_regressor_Predictive_Model.joblib` (tracked) | NumPy, pandas, scikit-learn, Joblib | Passed in prior local container integration | Present in Git; source/redistribution permission not independently established in this audit |
| PSI-10 Tad regressor | `POST /api/v1/predictions/psi10/adiabatic-flame-temperature` | `models/psi10/inverse_flame_tad_predictor.joblib` (tracked) | NumPy, pandas, scikit-learn, Joblib | Passed in prior local container integration | Present in Git; source/redistribution permission not independently established in this audit |
| PSI-10 Zst regressor | `POST /api/v1/predictions/psi10/stoichiometric-mixture-fraction` | `models/psi10/inverse_flame_zst_predictor.joblib` (tracked) | NumPy, pandas, scikit-learn, Joblib | Passed in prior local container integration | Present in Git; source/redistribution permission not independently established in this audit |
| PSI-10 pressure-rise regressor | `POST /api/v1/predictions/psi10/pressure-rise` | `models/psi10/inverse_flame_pressure_rise_predictor.joblib` (tracked) | NumPy, pandas, scikit-learn, Joblib | Passed in prior local container integration | Present in Git; source/redistribution permission not independently established in this audit |
| PSI-159 ignition classifier | `POST /api/v1/predictions/psi159/ignition` | `models/psi159/PSI_159_ignition_classifier_Predictive_Model.joblib` (tracked) | NumPy, pandas, scikit-learn, Joblib | Passed in prior local container integration | Present in Git; source/redistribution permission not independently established in this audit |
| PSI-107 fuel classifier | `POST /api/v1/predictions/psi107/fuel-classification` | `models/psi107/fuel_classifier_pipeline.joblib` (ignored locally; not tracked) | PyTorch CPU, NumPy, pandas, Joblib | Passed in local Docker integration | Local inference passed; public redistribution is not cleared; provide through an approved private artifact path |
| PSI-69 flame outcome classifier | `POST /api/v1/predictions/psi69/flame-outcome` | `models/psi69/dnn_cls_model.keras` and `dnn_cls_preprocessor.joblib` (ignored locally; not tracked) | TensorFlow/Keras, NumPy, pandas, Joblib | Passed in local Docker integration | Local inference passed; public redistribution is not cleared; provide through an approved private artifact path |

The prior Docker integration report records actual inference for all seven models in the existing local container. A new clean image was not built during this final audit. The local Python version is 3.9.6; the Docker image uses Python 3.11 because the current `google-genai` SDK requires Python 3.10 or newer. All model and dataset scientific provenance remains unverified; predictions are experimental and are not safety assessments. The five tracked model files are available in Git, but their redistribution permission was not independently established in this review.

The application also requires `data/raw/psi_full_dataset.csv` and `data/raw/psi_investigations_dataset.csv` for existing mission-intelligence routes. They are included. The larger external PSI-159 research CSV files are excluded because runtime code does not load them.

Both runtime CSVs are tracked demo fixtures, not a verified NASA research index. Their presence makes the mission endpoints buildable but does not establish source rights, NASA provenance, or research completeness.

## Local environment and Gemini

Use Python 3.11 for a clean local environment. `requirements.txt` pins direct cross-platform dependencies and `requirements-dev.txt` adds the test runner. `requirements.lock` is the complete Linux CPU Docker dependency snapshot; it includes the CPU-specific PyTorch wheel and is installed by the Dockerfile:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

Set `GEMINI_API_KEY` in the local `.env` for AI features. Keep the key blank in `.env.example`; `.env` is ignored by Git. Never place the key in frontend code, Docker build arguments, source files, or logs. `GEMINI_MODEL` defaults to the stable `gemini-3.8-flash` identifier, `GEMINI_TIMEOUT_MS` defaults to 25000, and the per-process Gemini request limit defaults to 30 calls per minute. Gemini is initialized lazily, so missing credentials do not prevent the other API routes from starting; AI routes return 503 until configured.

The Google GenAI SDK and model are selected from Google's official documentation: [SDK library guide](https://ai.google.dev/gemini-api/docs/libraries), [model list](https://ai.google.dev/gemini-api/docs/models), and [API-key security guidance](https://ai.google.dev/gemini-api/docs/api-key). The backend accepts only `gemini-3.8-flash`, currently listed with free standard input/output on the [Gemini API pricing page](https://ai.google.dev/gemini-api/docs/pricing). It rejects every other `GEMINI_MODEL` value and has no paid-model fallback. Google's [billing tiers](https://ai.google.dev/gemini-api/docs/billing) are project/account settings: the same model has paid-tier prices, so the app cannot guarantee zero charges if the API key belongs to a paid project. Use a Free Tier project to keep token usage free and monitor that project. The client makes one SDK attempt with a configurable 25-second HTTP timeout by default; rate limits and timeouts map to 429 and 504.

## AI endpoints

The existing `POST /api/v1/assistant/query` is now the grounded research chat route. It accepts a natural-language `question` and optional `investigation_ids`; if no IDs are supplied, a simple lexical search retrieves matching local fixture records. New endpoints are:

- `POST /api/v1/ai/research/summarize` — summarize one available investigation record under objective, method, conditions, findings, combustion parameters, mission relevance, and limitations.
- `POST /api/v1/ai/mission/explain` — explain the existing mission relevance ranking while returning the original numerical results unchanged.
- `POST /api/v1/ai/mission/report` — create a readable narrative from the existing structured mission report and include that structured report in the response.
- `POST /api/v1/ai/research/gaps/explain` — explain the gaps returned by the existing fixture-scoped gap logic.

Request examples:

```json
POST /api/v1/assistant/query
{"question":"What does the PSI-69 fixture report?","investigation_ids":["PSI-69"]}

POST /api/v1/ai/research/summarize
{"investigation_id":"PSI-69"}

POST /api/v1/ai/mission/explain
{"scenario":{"mission_environment":"lunar_habitat","oxygen_concentration_pct":21},"research_question":"flame spread","limit":5}

POST /api/v1/ai/mission/report
{"scenario":{"mission_environment":"space_station"},"research_question":"combustion","limit":5}

POST /api/v1/ai/research/gaps/explain
{"mission_environment":"mars_habitat","gravity_condition":"partial gravity"}
```

Generated text routes return an object with `content` (or `answer` / `report` / `explanation`), backend-supplied `sources`, `limitations`, and a status. Mission explanation/report/gap responses also include the original structured numeric results.

These features use only the repository's local CSV records. The CSVs are demo fixtures, not a verified NASA research index or full research papers. The API attaches backend-generated source IDs and unverified provenance; Gemini is instructed not to invent missing measurements, source links, or research findings. A missing context result does not call Gemini. Generated recommendations are limited to research review and data collection, not operational safety advice.

Provider 429 responses map to HTTP 429 with `Retry-After`; timeouts map to 504; other provider failures map to 502; missing or rejected credentials/model settings map to 503. The 30-per-minute in-memory limit is per process, not shared across replicas. Add a shared gateway/platform limit before scaling or exposing this endpoint to broad public traffic.

## Docker

The image uses Python 3.11, installs CPU PyTorch, runs as a non-root user, and serves Uvicorn without reload mode. Render's `PORT` is honored; local Docker defaults to port 8001. The Dockerfile copies an explicit allowlist of five Git-tracked model artifacts and the two runtime CSV fixtures. `.dockerignore` excludes the unresolved PSI-107 and PSI-69 files from the build context, and the Dockerfile does not copy them. Its early artifact guard checks all seven models and both CSVs and lists every missing path. A standard Git checkout therefore fails clearly for `models/psi107/fuel_classifier_pipeline.joblib`, `models/psi69/dnn_cls_model.keras`, and `models/psi69/dnn_cls_preprocessor.joblib`. This is an intentional release gate; no complete seven-model image is currently buildable from the public checkout. `requirements.lock` pins the runtime versions; the Python base tag and Debian package repository are still mutable, so bit-for-bit image reproduction would additionally require a pinned base-image digest and apt snapshot.

```bash
docker build -t pyronull-backend .
docker run --rm -p 8001:8001 --env-file .env -e PORT=8001 pyronull-backend
```

After artifact rights and delivery are approved, build and test with the commands above, then use the documented request bodies in `README.md`. Do not pass credentials as build arguments. The three PSI-107/PSI-69 files are excluded from both Git and Docker pending written redistribution permission; the checked-in Dockerfile allowlist excludes them too. A GitHub checkout will not contain them and will fail the early artifact check. A private artifact store does not itself establish a right to use or redistribute them. Once both rights and the destination are approved, a separate reviewed change must update the Docker context and copy allowlist. The private-store workflow and clearance checklist are in [MODEL_ARTIFACT_PROVENANCE_CHECKLIST.md](MODEL_ARTIFACT_PROVENANCE_CHECKLIST.md). Any image containing artifacts without public redistribution rights must remain private and only be published through an approved private registry.

## Render preparation

No Render Blueprint is checked into this repository. This section is a teammate handoff only; no Render service was created or deployed. **Current release status is NO-GO for the complete seven-model service:** the Git checkout does not contain the three ignored PSI-107/PSI-69 artifacts, and their provenance and redistribution permission are unresolved. Do not bypass the Docker artifact guard or claim all seven prediction endpoints are deployable until the missing artifacts are supplied through a permitted path and a clean image passes verification.

### Render environment variables

Set values in the Render service's Environment settings. Store `GEMINI_API_KEY` using Render's secret environment-variable controls. Never put secret values in Git, `.env.example`, Docker build arguments, or image layers.

| Variable | Required status | Render handoff |
| --- | --- | --- |
| `PORT` | Platform-provided; optional locally (defaults to `8001`) | Render supplies this for the web service. Dockerfile and Uvicorn bind to `0.0.0.0:$PORT`. |
| `GEMINI_API_KEY` | Optional for backend startup; required to use Gemini endpoints | Add as a Render secret. No real key belongs in repository files. |
| `GEMINI_MODEL` | Optional; defaults to `gemini-3.8-flash` | Only this allowlisted model is accepted. The app cannot confirm the Google project's billing tier. |
| `GEMINI_TIMEOUT_MS` | Optional; defaults to `25000` | Allowed range is 1000–120000 milliseconds. |
| `GEMINI_REQUESTS_PER_MINUTE` | Optional; defaults to `30` | Per-process limit, not shared across Render instances. |
| `CORS_ALLOWED_ORIGINS` | Optional for API-only use; set for browser clients | Comma-separated exact frontend origins, including scheme. Do not use a broad wildcard for production. |
| `APP_ENV` | Not required; currently unused by application code | Present in `.env.example` for compatibility only; it does not switch runtime behavior. |
| `API_V1_PREFIX` | Not required; currently unused by application code | Present in `.env.example` for compatibility only; route prefix is currently `/api/v1`. |

The Gemini model is free-tier eligible, but zero charges cannot be guaranteed by code: confirm the API key's Google project remains on the Free Tier and monitor usage. Do not attach a paid-tier key if free-tier use is required.

Choose either deployment method in the Render Dashboard:

### Docker service

- Runtime: Docker; Dockerfile path: `./Dockerfile`.
- Health check path: `/health`.
- Start command: use the Dockerfile `CMD`; it binds to `0.0.0.0` and reads Render's `PORT`.
- Environment: set `GEMINI_API_KEY` as a secret, set `GEMINI_MODEL=gemini-3.8-flash`, and set `CORS_ALLOWED_ORIGINS` to exact frontend origins. Do not set secrets as Docker build arguments.
- Do not include PSI-107/PSI-69 artifacts until source, use, and destination permissions are documented. They are excluded from the current Git and Docker contexts and the seven-model guard will fail without them.
- Do not make the resulting image public while it embeds artifacts whose redistribution rights are unresolved.

### Native Python service

- Set the service's Python version to 3.11 (the SDK requires 3.10+) and install the CPU PyTorch wheel first from `https://download.pytorch.org/whl/cpu`, then install `requirements.lock` with `pip install --no-deps -r requirements.lock` and run `pip check`. Prefer the Docker service for parity with the tested system packages and CPU dependency set.
- Use start command `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- Set health check path `/health`, plus the same runtime environment variables described above.
- Ensure the deployment checkout contains all required model artifacts and the two `data/raw/` CSVs.

Both modes install TensorFlow and PyTorch and therefore need more memory and build time than a small pure-Python API. Each worker process caches its own loaded models; adding workers or instances multiplies memory use. See [Render Docker deployment](https://render.com/docs/docker), [Render health checks](https://render.com/docs/health-checks), and [Render environment variables](https://render.com/docs/configure-environment-variables). A successful image build does not verify Render credentials, service settings, model availability, or health-check behavior.

Before the teammate builds, obtain written artifact permissions and approve a delivery mechanism. A private store is not an alternative to rights clearance. After approval, review a code/configuration change that adds the exact approved files to the Docker context and Dockerfile allowlist, fetch immutable artifact versions into a private build context, verify recorded hashes, and keep any image with private-only assets in a private registry. If no authorized artifacts are available, the complete service remains blocked; a limited product release would require a separate reviewed decision and explicit endpoint behavior changes. Do not substitute dummy files or models.

## Verification and troubleshooting

```bash
python -m pytest -q
python -m pip check
git diff --check
docker build -t pyronull-backend .
docker run --rm -p 8001:8001 --env-file .env -e PORT=8001 pyronull-backend
```

Common failures:

- **AI route returns 503:** configure `GEMINI_API_KEY`, check that the key/model is valid, and restart/redeploy.
- **AI route returns 429:** respect `Retry-After`, review Gemini quota, and configure a shared rate limit for multiple instances.
- **Prediction route returns 503:** a model artifact is missing, altered, or incompatible; inspect deployment artifacts and preserve the repository-relative directory structure.
- **Mission route returns 503:** one of the two required `data/raw/` CSVs is missing or malformed.
- **Browser CORS error:** set `CORS_ALLOWED_ORIGINS` to exact origins including scheme and optional port.
- **Container fails to become healthy:** inspect logs, confirm `PORT` binding and `/health`, then test the container locally.

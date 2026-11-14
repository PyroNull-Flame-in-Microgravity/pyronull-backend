# PyroNull Backend — FIREWISE AI

FastAPI backend for the PyroNull team's NASA Space Apps Challenge project, **Flame in Freefall**.

The product is an evidence-linked research dashboard to discover, compare and interpret NASA microgravity combustion investigations relevant to spacecraft fire prevention, detection and suppression. It is **not a certified spacecraft fire-risk predictor**.

## Current status
This is the initial API skeleton. Included investigation entries are demo placeholders, not verified NASA PSI records. Replace them only after inspecting live PSI records and recording exact identifiers, source URLs, files and fields.

## API
Base path: `/api/v1`

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/investigations` | List/filter investigations |
| GET | `/investigations/{id}` | Investigation detail |
| POST | `/compare` | Compare investigations/runs |
| POST | `/assistant/query` | Grounded research Q&A (not implemented yet) |
| POST | `/rank` | Research relevance ranking (not implemented yet) |

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

## Next steps
1. Agree response contracts with frontend and AI/ML teammates.
2. Replace placeholders with a verified NASA PSI source inventory.
3. Implement source-aware parsing and metadata storage.
4. Implement scientifically valid comparisons.
5. Add document retrieval/RAG with page/section citations.
6. Implement transparent relevance scoring and evaluation.
7. Add tests and deployment configuration.

Never commit credentials, `.env`, private data or large raw NASA files. Commit `.env.example` instead.

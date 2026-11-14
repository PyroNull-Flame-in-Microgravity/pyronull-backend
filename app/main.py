import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router


DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://localhost:5173"


def _configured_cors_origins(environ=None):
    """Read comma-separated browser origins from CORS_ALLOWED_ORIGINS."""
    values = os.environ if environ is None else environ
    configured = values.get("CORS_ALLOWED_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]


app = FastAPI(title="PyroNull / FIREWISE AI Backend", version="0.1.0",
              description="Evidence-linked NASA microgravity combustion research APIs.")
app.add_middleware(CORSMiddleware, allow_origins=_configured_cors_origins(),
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(api_router, prefix="/api/v1")

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "pyronull-backend"}

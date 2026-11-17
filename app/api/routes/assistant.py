from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.schemas.gemini import AssistantResponse
from app.services.gemini_assistant import (
    DATA_LIMITATIONS,
    GeminiConfigurationError,
    GeminiProviderError,
    GeminiRateLimitError,
    GeminiTimeoutError,
    answer_question,
)
from app.services.mission_intelligence import DatasetLoadError

router = APIRouter()


class AssistantQuery(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    investigation_ids: list[str] = Field(default_factory=list, max_length=20)


def _raise_ai_http_error(exc):
    if isinstance(exc, GeminiConfigurationError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, GeminiRateLimitError):
        raise HTTPException(
            status_code=429,
            detail="AI request limit reached. Try again later.",
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    if isinstance(exc, GeminiTimeoutError):
        raise HTTPException(status_code=504, detail="Gemini request timed out.") from exc
    if isinstance(exc, GeminiProviderError):
        raise HTTPException(status_code=502, detail="Gemini could not complete the request.") from exc
    raise exc


@router.post("/assistant/query", response_model=AssistantResponse)
def ask(payload: AssistantQuery):
    try:
        answer, sources, status = answer_question(payload.question, payload.investigation_ids)
    except (GeminiConfigurationError, GeminiRateLimitError, GeminiTimeoutError, GeminiProviderError) as exc:
        _raise_ai_http_error(exc)
    except DatasetLoadError as exc:
        raise HTTPException(status_code=503, detail="Research fixture data is unavailable.") from exc
    limitations = list(DATA_LIMITATIONS)
    if status == "NO_CONTEXT":
        limitations.append("No matching investigation records were found in the supplied fixture data; Gemini was not called.")
    return AssistantResponse(answer=answer, sources=sources, limitations=limitations, status=status)

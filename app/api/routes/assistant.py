from fastapi import APIRouter
from pydantic import BaseModel, Field
router = APIRouter()
class AssistantQuery(BaseModel):
    question: str = Field(min_length=3)
    investigation_ids: list[str] = []
@router.post("/assistant/query")
def ask(payload: AssistantQuery):
    return {"answer": None, "sources": [],
            "limitations": ["Not connected to a verified NASA document index yet."],
            "status": "NOT_IMPLEMENTED"}

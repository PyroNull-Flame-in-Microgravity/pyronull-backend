from fastapi import APIRouter
from pydantic import BaseModel, Field
router = APIRouter()
class CompareRequest(BaseModel):
    investigation_ids: list[str] = Field(min_length=2)
@router.post("/compare")
def compare(payload: CompareRequest):
    return {"requested_ids": payload.investigation_ids, "common_fields": [], "differences": [],
            "warnings": ["Not connected to verified NASA source data.",
                         "Do not treat unlike experiments as controlled head-to-head trials."],
            "status": "NOT_IMPLEMENTED"}

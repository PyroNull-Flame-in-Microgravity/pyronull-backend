from fastapi import APIRouter
from pydantic import BaseModel, Field
router = APIRouter()
class RankRequest(BaseModel):
    research_question: str = Field(min_length=3)
@router.post("/rank")
def rank(payload: RankRequest):
    return {"research_question": payload.research_question, "ranked_items": [],
            "scoring_method": {"objective_match": 0.40, "condition_similarity": 0.25,
                               "evidence_completeness": 0.20, "mission_context_fit": 0.15},
            "score_label": "RESEARCH_RELEVANCE_NOT_FIRE_RISK", "status": "NOT_IMPLEMENTED"}

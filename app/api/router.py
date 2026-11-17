from fastapi import APIRouter
from app.api.routes import investigations, compare, assistant, rank, mission_intelligence, predictions, ai
api_router = APIRouter()
api_router.include_router(investigations.router, tags=["Investigations"])
api_router.include_router(compare.router, tags=["Comparison"])
api_router.include_router(assistant.router, tags=["Research Assistant"])
api_router.include_router(rank.router, tags=["Research Relevance"])
api_router.include_router(mission_intelligence.router, tags=["Mission Intelligence"])
api_router.include_router(predictions.router, tags=["Predictions"])
api_router.include_router(ai.router, tags=["AI Research Assistant"])

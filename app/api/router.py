from fastapi import APIRouter
from app.api.routes import investigations, compare, assistant, rank
api_router = APIRouter()
api_router.include_router(investigations.router, tags=["Investigations"])
api_router.include_router(compare.router, tags=["Comparison"])
api_router.include_router(assistant.router, tags=["Research Assistant"])
api_router.include_router(rank.router, tags=["Research Relevance"])

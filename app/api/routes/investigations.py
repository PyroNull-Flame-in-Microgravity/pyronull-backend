
from typing import Optional

from fastapi import APIRouter, HTTPException
from app.data.sample_data import INVESTIGATIONS

router = APIRouter()


@router.get("/investigations")
def list_investigations(
    domain: Optional[str] = None,
    search: Optional[str] = None
):
    items = INVESTIGATIONS

    if domain:
        items = [
            x for x in items
            if domain.lower() in [d.lower() for d in x["domains"]]
        ]

    if search:
        q = search.lower()
        items = [
            x for x in items
            if q in (x["title"] + " " + x["summary"]).lower()
        ]

    return {
        "items": items,
        "total": len(items),
        "data_status": "DEMO_PLACEHOLDERS_NOT_VERIFIED_NASA_RECORDS"
    }


@router.get("/investigations/{investigation_id}")
def get_investigation(investigation_id: str):
    for item in INVESTIGATIONS:
        if item["id"] == investigation_id:
            return {
                "investigation": item,
                "data_status": "DEMO_PLACEHOLDER_NOT_VERIFIED_NASA_RECORD"
            }

    raise HTTPException(
        status_code=404,
        detail="Investigation not found"
    )

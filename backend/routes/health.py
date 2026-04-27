"""GET /health — liveness probe used by Electron's SidecarManager."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}

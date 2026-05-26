from fastapi import APIRouter, Depends, Request
from app.api.schemas import HealthResponse, ReadinessResponse
from app.config import get_settings
import psycopg2

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse, summary="Liveness check")
async def liveness() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.app_version
    )


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness check")
async def readiness(request: Request) -> ReadinessResponse:
    settings = get_settings()

    # Check Qdrant Connection
    vector_store = request.app.state.vector_store
    qdrant_connected = vector_store.health_check()
    collection_info = vector_store.get_collection_info()

    # Check Postgres Connection
    postgres_connected = False
    try:
        conn = psycopg2.connect(settings.database_url)
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
        conn.close()
        postgres_connected = True
    except Exception:
        pass

    status = "ready" if (qdrant_connected and postgres_connected) else "degraded"

    return ReadinessResponse(
        status=status,
        qdrant_connected=qdrant_connected,
        postgres_connected=postgres_connected,
        collection_info=collection_info
    )

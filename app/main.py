from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from app.api.routes import chat, documents, health, memory, sql, mutation, cache
from app.config import get_settings
from app.core.csrag_engine import CSRAGEngine
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger, setup_logging

settings = get_settings()
__version__ = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    logger = get_logger(__name__)
    logger.info(f"Starting {settings.app_name} v{__version__}")

    logger.info("Initializing VectorStoreService (Qdrant)...")
    app.state.vector_store = VectorStoreService()
    logger.info("VectorStoreService ready")

    logger.info("Connecting AsyncPostgresStore (LTM)...")
    # Using database_url as the unified Postgres connection string
    async with AsyncPostgresStore.from_conn_string(settings.database_url) as store:
        await store.setup()
        app.state.store = store
        logger.info("AsyncPostgresStore (LTM) ready")

        logger.info("Connecting AsyncPostgresSaver (STM checkpointer)...")
        async with AsyncPostgresSaver.from_conn_string(settings.database_url) as checkpointer:
            await checkpointer.setup()
            app.state.checkpointer = checkpointer
            logger.info("AsyncPostgresSaver (STM checkpointer) ready")

            logger.info("Compiling IDOP Graph Engine...")
            app.state.engine = CSRAGEngine(
                vector_store=app.state.vector_store,
                store=store,
                checkpointer=checkpointer,
            )
            logger.info("IDOP Engine ready — all services online")

            yield

    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.app_name,
    description="""
## IDOP — Intelligent Data Operations Platform

An enterprise-grade platform combining text-to-SQL, transactional safe mutations, and multi-source corrective self-reflective RAG:

- **Feature 1**: NL-to-SQL query generation with LLM semantic auditing and safety constraints validation.
- **Feature 2**: Rollback-safe bulk Excel/CSV document mutations with configuration JSON business guardrails.
- **Feature 3**: Advanced RAG query pipeline using HyDE, Reciprocal Rank Fusion, CRAG relevance gates, and Context Enrichment Windowing.
- **Memory**: STM summaries and long-term user profile facts personalization.
- **Caching**: Dual caching utilizing Upstash Redis (queries/SQL results) and S3 storage (ingested document chunks).
    """,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

_raw_origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
allowed_origins: list[str] = _raw_origins if _raw_origins != ["*"] else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(memory.router)
app.include_router(sql.router)
app.include_router(mutation.router)
app.include_router(cache.router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": settings.app_name,
        "version": __version__,
        "docs": "/docs",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger = get_logger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )

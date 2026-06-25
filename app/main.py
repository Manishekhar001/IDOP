import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from contextlib import AsyncExitStack, asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: E402
from langgraph.store.postgres.aio import AsyncPostgresStore  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from slowapi import Limiter, _rate_limit_exceeded_handler  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402
from slowapi.util import get_remote_address  # noqa: E402

from app.api.auth import create_users_table  # noqa: E402
from app.api.routes import (  # noqa: E402
    auth_routes,
    cache,
    chat,
    documents,
    health,
    memory,
    mutation,
    sql,
)
from app.config import __version__, get_settings  # noqa: E402
from app.core.csrag_engine import CSRAGEngine  # noqa: E402
from app.core.vector_store import VectorStoreService  # noqa: E402
from app.utils.logger import get_logger, setup_logging  # noqa: E402

settings = get_settings()

_exit_stack = AsyncExitStack()


async def _retry_init(
    factory, name: str, max_retries: int = 3, initial_delay: float = 1.0
):
    """
    Generic async retry wrapper for Postgres-backed resource initialization.

    Freshly-restarted Postgres containers sometimes close the first few
    connections during background recovery. Retrying the full connection
    + migration cycle resolves this transient psycopg.OperationalError.

    The factory is responsible for its own cleanup on failure.
    """
    logger = get_logger(__name__)
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resource = await factory()
            logger.info(f"{name} ready on attempt {attempt}")
            return resource
        except Exception as e:
            last_exc = e
            logger.warning(
                f"{name} init attempt {attempt}/{max_retries} failed: "
                f"{type(e).__name__}: {e}"
            )
            if attempt < max_retries:
                delay = initial_delay * (2 ** (attempt - 1))
                logger.info(f"Retrying {name} init in {delay}s...")
                await asyncio.sleep(delay)
    logger.critical(
        f"All {max_retries} {name} init attempts failed. Last error: {last_exc}"
    )
    raise last_exc


async def _connect_pg_resource(cls):
    """
    Create a Postgres resource and run migrations.
    """
    cm = cls.from_conn_string(settings.database_url)
    resource = await _exit_stack.enter_async_context(cm)
    await resource.setup()
    return resource


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    logger = get_logger(__name__)
    logger.info(f"Starting {settings.app_name} v{__version__}")

    # ── ONE-TIME: delete stale collection so sparse vectors are regenerated ──
    # The old SparseVectorService used Python hash() which is non-deterministic
    # across processes.  Delete the collection so all vectors (dense + sparse)
    # are re-ingested with the new deterministic fastembed BM25 model.
    # TODO: Remove this block after the first successful deploy.
    logger.warning(
        "ONE-TIME: Deleting Qdrant collection to purge stale hash()-based sparse vectors"
    )
    try:
        _temp_client = QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key
        )
        _temp_client.delete_collection(collection_name=settings.collection_name)
        logger.info(
            f"ONE-TIME: Deleted collection '{settings.collection_name}' — will be recreated by VectorStoreService"
        )
    except Exception as _del_err:
        logger.info(
            f"ONE-TIME: Collection delete skipped (may not exist yet): {_del_err}"
        )
    # ── END ONE-TIME ──────────────────────────────────────────────────────────

    logger.info("Initializing VectorStoreService (Qdrant)...")
    app.state.vector_store = VectorStoreService()
    logger.info("VectorStoreService ready")

    # Create users table for JWT authentication
    logger.info("Ensuring idop_users table exists...")
    create_users_table()

    # ---- D3: ensure all business tables exist at startup ----
    logger.info("Ensuring business tables exist...")
    try:
        import psycopg2

        _startup_conn = psycopg2.connect(
            settings.supabase_db_url or settings.database_url
        )
        try:
            from app.core.approval_gate import approval_gate, mutation_approval_gate
            from app.core.audit_logger import AuditLogger
            from app.services.pending_store import pending_mutations, pending_queries

            approval_gate._ensure_table(_startup_conn)
            mutation_approval_gate._ensure_table(_startup_conn)
            AuditLogger().ensure_table(_startup_conn)
            pending_queries._ensure_table(_startup_conn)
            pending_mutations._ensure_table(_startup_conn)
            logger.info("All business tables verified")
        except Exception as inner_exc:
            logger.warning("Could not run business tables verification: %s", inner_exc)
        finally:
            _startup_conn.close()
    except Exception as exc:
        logger.warning("Could not verify business tables at startup: %s", exc)

    # Run both Postgres inits in parallel — they're independent and this cuts
    # worst-case startup from ~34s to ~17s (exponential backoff: 1s, 2s, 4s, 8s).
    # EC2 cold start (t2.micro) can take up to 115s for Postgres to be healthy.
    # The CD pipeline health check waits 210s.  Our retry window must cover that.
    # Exponential backoff: 3, 6, 12, 24, 48, 96, 192 = 381s total.
    # This comfortably exceeds the 210s compose healthcheck timeout.
    logger.info("Connecting AsyncPostgresStore (LTM) and AsyncPostgresSaver...")
    store_task = _retry_init(
        lambda: _connect_pg_resource(AsyncPostgresStore),
        "AsyncPostgresStore (LTM)",
        max_retries=8,
        initial_delay=3.0,
    )
    checkpointer_task = _retry_init(
        lambda: _connect_pg_resource(AsyncPostgresSaver),
        "AsyncPostgresSaver (checkpointer)",
        max_retries=8,
        initial_delay=3.0,
    )
    store, checkpointer = await asyncio.gather(store_task, checkpointer_task)
    app.state.store = store
    app.state.checkpointer = checkpointer

    logger.info("Compiling IDOP Graph Engine...")
    app.state.engine = CSRAGEngine(
        vector_store=app.state.vector_store, store=store, checkpointer=checkpointer
    )
    logger.info("IDOP Engine ready — all services online")

    yield

    logger.info("Shutting down services...")
    await _exit_stack.aclose()
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

if allowed_origins == ["*"]:
    # Wildcard origin — credentials MUST be False per CORS spec
    # https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS/Errors/CORSNotSupportingCredentials
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ── Rate Limiting ─────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(auth_routes.router)
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(memory.router)
app.include_router(sql.router)
app.include_router(mutation.router)
app.include_router(cache.router)


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with welcome message and navigation links.

    Returns a brief introduction to the IDOP platform with links to
    interactive API documentation and health monitoring.

    Returns:
        dict: Platform welcome message with service name, version, and documentation links.
    """
    return {
        "service": settings.app_name,
        "version": __version__,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger = get_logger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500, content={"error": "Internal Server Error", "message": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app", host=settings.api_host, port=settings.api_port, reload=True
    )

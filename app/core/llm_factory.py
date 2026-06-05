"""
Shared LLM Factory — uses LiteLLM Router for multi-provider, multi-key load balancing.

Supports:
  - Multiple Groq API keys with automatic failover (primary key hits rate limit → next key)
  - Multiple providers (OpenAI, Groq, Together, etc.) via LiteLLM Router
  - Fallback to direct ChatOpenAI / ChatGroq if LiteLLM is not installed

All components (CRAG, SRAG, HyDE, RAGAS, graph nodes, memory services) import
`get_chat_llm()` from this module instead of instantiating ChatOpenAI directly.

Usage:
    from app.core.llm_factory import get_chat_llm

    llm = get_chat_llm()
    result = await llm.ainvoke(...)

    # Structured output
    chain = prompt | llm.with_structured_output(MyModel)

Configuration (in .env):
    LLM_PROVIDER=litellm       # "litellm", "groq", or "openai" (default: "openai")
    LLM_MODEL=gpt-4o            # Model name (for OpenAI) or model group (for LiteLLM)
    LLM_TEMPERATURE=0.0

    # For LiteLLM with multiple Groq keys:
    LLM_PROVIDER=litellm
    LLM_MODEL=llama-3.3-70b-versatile
    GROQ_API_KEY_1=gsk_abc...
    GROQ_API_KEY_2=gsk_def...
    GROQ_API_KEY_3=gsk_ghi...

    # For direct Groq (single key):
    LLM_PROVIDER=groq
    GROQ_API_KEY=gsk_abc...
    LLM_MODEL=llama-3.3-70b-versatile

    # For OpenAI:
    LLM_PROVIDER=openai
    OPENAI_API_KEY=sk-...
    LLM_MODEL=gpt-4o
"""

import logging
from functools import lru_cache
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import get_settings

logger = logging.getLogger("idop_app.llm_factory")

# ── Groq model name aliases (shorthand → full API name) ──────────────
GROQ_MODEL_ALIASES: dict[str, str] = {
    "llama-3.3-70b": "llama-3.3-70b-versatile",
    "llama-3.1-8b": "llama-3.1-8b-instant",
    "llama-3.2-3b": "llama-3.2-3b-preview",
    "mixtral-8x7b": "mixtral-8x7b-32768",
    "gemma2-9b": "gemma2-9b-it",
    "gemma-2-9b": "gemma2-9b-it",
}


def _resolve_groq_model(model: str) -> str:
    """Resolve shorthand aliases to full Groq API model names."""
    model = model.strip()
    if model.lower() in GROQ_MODEL_ALIASES:
        resolved = GROQ_MODEL_ALIASES[model.lower()]
        logger.debug(f"Resolved Groq model alias '{model}' -> '{resolved}'")
        return resolved
    return model


def _build_litellm_router() -> Any:
    """Build a LiteLLM Router with all configured API keys and providers.

    Creates:
      - One deployment per Groq API key (groq/llama-3.3-70b-versatile)
      - One OpenAI fallback deployment (openai/gpt-4o) as last resort

    The Router automatically load-balances, detects 429 rate limits,
    cools down failed deployments, and falls back to healthy ones.
    """
    from litellm import Router

    settings = get_settings()
    model = _resolve_groq_model(settings.llm_model)
    groq_model = f"groq/{model}"
    model_group = "idop-llm"  # single group so Router picks best available

    # Collect all Groq keys
    groq_keys = settings.groq_api_keys
    if not groq_keys:
        logger.warning("No Groq API keys configured for LiteLLM Router — falling back")
        return None

    # Build deployment list — Groq keys first (primary), OpenAI last (fallback)
    deployments = []
    for i, key in enumerate(groq_keys):
        deployments.append({
            "model": groq_model,
            "api_key": key,
            "rpm": 30,  # Groq free tier: 30 req/min per key
        })
        logger.info(f"  Groq deployment {i+1}: groq/{model}")

    # Add OpenAI as the final fallback
    openai_key = settings.openai_api_key
    if openai_key and openai_key.strip():
        deployments.append({
            "model": "openai/gpt-4o",
            "api_key": openai_key.strip(),
            "rpm": 500,  # Higher RPM since it's the final fallback
        })
        logger.info(f"  OpenAI fallback: openai/gpt-4o")

    # Configure the Router with cooldown and retry settings
    model_list = [
        {
            "model_name": model_group,  # all under one model group
            "litellm_params": dep,
        }
        for dep in deployments
    ]

    router = Router(
        model_list=model_list,
        num_retries=2,              # Retries per request on failure
        retry_after=5.0,            # Seconds between retries
        allowed_fails=3,            # Max consecutive fails before cooldown
        cooldown_time=60.0,         # Seconds to cool down a failed deployment
        routing_strategy="latency-based-routing",
    )

    logger.info(
        f"LiteLLM Router initialized with {len(deployments)} total deployments "
        f"({len(groq_keys)} Groq + {1 if openai_key else 0} OpenAI) "
        f"for model group '{model_group}'"
    )
    return router


@lru_cache
def get_chat_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> BaseChatModel:
    """
    Return a configured chat LLM based on settings.

    Priority:
      1. If LLM_PROVIDER=litellm → use ChatLiteLLMRouter (multi-key load balancing)
      2. If LLM_PROVIDER=groq → use ChatGroq (single key)
      3. Default → use ChatOpenAI

    The result is cached via @lru_cache so the same LLM instance is reused.
    """
    settings = get_settings()
    provider = settings.llm_provider
    resolved_model = model or settings.llm_model
    resolved_temp = temperature if temperature is not None else settings.llm_temperature

    # ── Option 1: LiteLLM Router (multi-key load balancing) ──────────────
    if provider == "litellm":
        try:
            from langchain_litellm import ChatLiteLLMRouter

            router = _build_litellm_router()
            if router is not None:
                logger.info(
                    f"Creating ChatLiteLLMRouter: model_group=idop-llm, "
                    f"temperature={resolved_temp}"
                )
                return ChatLiteLLMRouter(
                    model="idop-llm",
                    temperature=resolved_temp,
                    router=router,
                )
            # If router is None (no keys), fall through to next option
        except ImportError:
            logger.warning(
                "langchain-litellm not installed. Run: uv pip install litellm langchain-litellm"
            )
            # Fall through to direct provider
        except Exception as e:
            logger.warning(f"LiteLLM Router initialization failed: {e}. Falling back.")
            # Fall through

    # ── Option 2: Direct Groq ────────────────────────────────────────────
    if provider in ("groq", "litellm"):  # litellm falls through here too
        groq_keys = settings.groq_api_keys
        if groq_keys:
            resolved_model = _resolve_groq_model(resolved_model)
            try:
                from langchain_groq import ChatGroq

                logger.info(
                    f"Creating ChatGroq: model={resolved_model}, "
                    f"temperature={resolved_temp} (using key #1)"
                )
                return ChatGroq(
                    model=resolved_model,
                    temperature=resolved_temp,
                    api_key=groq_keys[0],
                )
            except ImportError:
                logger.warning("langchain-groq not installed. Falling back to OpenAI.")
        else:
            logger.warning("No Groq keys configured.")

    # ── Option 3: OpenAI (default fallback) ─────────────────────────────
    from langchain_openai import ChatOpenAI

    logger.info(
        f"Creating OpenAI LLM: model={resolved_model}, temperature={resolved_temp}"
    )
    return ChatOpenAI(
        model=resolved_model,
        temperature=resolved_temp,
        api_key=settings.openai_api_key,
    )


@lru_cache
def get_memory_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> BaseChatModel:
    """
    Return a lighter/cheaper LLM for memory/classification tasks.

    Uses memory_llm_model setting if available, otherwise falls back
    to the primary LLM model.
    """
    settings = get_settings()
    resolved_model = (
        model
        or getattr(settings, "memory_llm_model", None)
        or settings.llm_model
    )
    resolved_temp = (
        temperature
        if temperature is not None
        else getattr(settings, "memory_llm_temperature", settings.llm_temperature)
    )
    return get_chat_llm(model=resolved_model, temperature=resolved_temp)

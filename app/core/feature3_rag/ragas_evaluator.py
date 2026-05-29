"""
Lightweight RAGAS-style evaluation service.
Computes answer_relevancy, faithfulness, and context_precision using the
existing OpenAI LLM — no extra library dependencies required.

Mirrors the pattern used by CRAG (evaluator.py) and SRAG (verifier.py).
"""

import logging
from functools import lru_cache
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import get_settings

logger = logging.getLogger("idop_app.ragas_evaluator")


# ──────────────────────────────────────────────
# Pydantic schemas for structured LLM output
# ──────────────────────────────────────────────


class RelevancyScore(BaseModel):
    """Answer relevancy score and analysis."""
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Answer relevancy score. 1.0 = perfectly addresses the question.",
    )
    reason: str = Field(
        ...,
        description="Short justification explaining the score.",
    )


class FaithfulnessScore(BaseModel):
    """Faithfulness score measuring factual consistency with context."""
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Faithfulness score. 1.0 = every claim in the answer is directly supported by the context.",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="List of claims in the answer that are NOT supported by the context.",
    )


class ContextPrecisionScore(BaseModel):
    """Context precision / relevance score."""
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Context precision score. 1.0 = all retrieved chunks are highly relevant to the question.",
    )
    num_relevant: int = Field(
        ...,
        description="Number of retrieved chunks judged relevant.",
    )
    num_total: int = Field(
        ...,
        description="Total number of chunks evaluated.",
    )


class RagasScores(BaseModel):
    """Aggregated RAGAS evaluation results."""
    answer_relevancy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Answer relevancy score — how well the answer addresses the question.",
    )
    answer_relevancy_reason: str = Field(
        ...,
        description="Short justification for the answer relevancy score.",
    )
    faithfulness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Faithfulness score — factual consistency between answer and retrieved context.",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the answer not supported by the retrieved context.",
    )
    context_precision: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Context precision — relevance of the retrieved document chunks to the question.",
    )
    context_relevant_count: int = Field(
        ...,
        description="Number of retrieved chunks judged relevant out of total.",
    )
    context_total_count: int = Field(
        ...,
        description="Total number of retrieved chunks evaluated.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer_relevancy": 0.95,
                    "answer_relevancy_reason": "Answer directly addresses the question with specific data from the database.",
                    "faithfulness": 0.88,
                    "unsupported_claims": [],
                    "context_precision": 0.82,
                    "context_relevant_count": 3,
                    "context_total_count": 4,
                }
            ]
        }
    }


# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

_RELEVANCY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict answer relevancy judge. Given a question and an answer, "
            "determine how relevant the answer is to the question on a scale of 0.0 to 1.0.\n\n"
            "Scoring guide:\n"
            "  1.0 — Answer perfectly and completely addresses the question.\n"
            "  0.7 — Answer is mostly relevant but missing some details.\n"
            "  0.5 — Answer is partially relevant (related but doesn't directly answer).\n"
            "  0.3 — Answer is marginally relevant (tangentially related).\n"
            "  0.0 — Answer is completely irrelevant or off-topic.\n\n"
            "Output JSON with 'score' (float) and 'reason' (string).",
        ),
        ("human", "Question: {question}\n\nAnswer:\n{answer}"),
    ]
)

_FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict factual consistency auditor. Given a context and an answer, "
            "determine how factually consistent the answer is with the provided context "
            "on a scale of 0.0 to 1.0.\n\n"
            "Scoring guide:\n"
            "  1.0 — Every claim in the answer is directly supported by the context.\n"
            "  0.7 — Most claims are supported, but some are implied rather than explicit.\n"
            "  0.5 — Some claims are supported, others are not.\n"
            "  0.3 — Most claims are not supported by the context.\n"
            "  0.0 — Answer contradicts the context or is entirely fabricated.\n\n"
            "Also list any unsupported claims as strings in 'unsupported_claims'.\n"
            "Output JSON with 'score' (float) and 'unsupported_claims' (list of strings).",
        ),
        ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
    ]
)

_CONTEXT_PRECISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval precision judge. Given a question and a list of "
            "retrieved document chunks, determine how relevant the retrieved chunks are "
            "to the question on a scale of 0.0 to 1.0.\n\n"
            "Scoring guide:\n"
            "  1.0 — All retrieved chunks are highly relevant and directly answer the question.\n"
            "  0.7 — Most chunks are relevant, some are tangentially related.\n"
            "  0.5 — About half the chunks are relevant.\n"
            "  0.3 — Few chunks are relevant.\n"
            "  0.0 — No retrieved chunks are relevant to the question.\n\n"
            "Also specify 'num_relevant' (how many of the chunks you consider relevant) "
            "and 'num_total' (total chunks evaluated).\n"
            "Output JSON with 'score' (float), 'num_relevant' (int), and 'num_total' (int).",
        ),
        ("human", "Question: {question}\n\nRetrieved Chunks:\n{chunks_text}"),
    ]
)


# ──────────────────────────────────────────────
# Service
# ──────────────────────────────────────────────


class RagasEvaluator:
    """Lightweight RAGAS evaluation service using OpenAI LLM."""

    def __init__(self) -> None:
        settings = get_settings()
        llm = ChatOpenAI(
            model=settings.memory_llm_model,
            temperature=0.0,
            api_key=settings.openai_api_key,
        )
        self._relevancy_chain = _RELEVANCY_PROMPT | llm.with_structured_output(RelevancyScore)
        self._faithfulness_chain = _FAITHFULNESS_PROMPT | llm.with_structured_output(FaithfulnessScore)
        self._precision_chain = _CONTEXT_PRECISION_PROMPT | llm.with_structured_output(ContextPrecisionScore)
        logger.info("RAGASEvaluator initialized")

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> Optional[RagasScores]:
        """
        Compute RAGAS-style metrics for a single Q/A pair.
        Returns a RagasScores object, or None if evaluation fails entirely.
        """
        if not answer:
            logger.warning("RAGAS evaluation skipped: empty answer")
            return None

        try:
            # 1. Answer Relevancy
            relevancy: RelevancyScore = await self._relevancy_chain.ainvoke(
                {"question": question, "answer": answer}
            )
            logger.debug(f"RAGAS answer_relevancy: {relevancy.score:.3f} — {relevancy.reason}")

            # 2. Faithfulness
            context_str = "\n\n---\n\n".join(contexts) if contexts else "(no context provided)"
            faithfulness: FaithfulnessScore = await self._faithfulness_chain.ainvoke(
                {"context": context_str, "answer": answer}
            )
            logger.debug(
                f"RAGAS faithfulness: {faithfulness.score:.3f} "
                f"({len(faithfulness.unsupported_claims)} unsupported claims)"
            )

            # 3. Context Precision
            chunks_text = "\n\n---\n\n".join(
                f"Chunk {i+1}: {c[:300]}"
                for i, c in enumerate(contexts[:10])
            ) if contexts else "(no chunks retrieved)"
            precision: ContextPrecisionScore = await self._precision_chain.ainvoke(
                {"question": question, "chunks_text": chunks_text}
            )
            logger.debug(
                f"RAGAS context_precision: {precision.score:.3f} "
                f"({precision.num_relevant}/{precision.num_total} relevant)"
            )

            return RagasScores(
                answer_relevancy=round(relevancy.score, 4),
                answer_relevancy_reason=relevancy.reason,
                faithfulness=round(faithfulness.score, 4),
                unsupported_claims=faithfulness.unsupported_claims,
                context_precision=round(precision.score, 4),
                context_relevant_count=precision.num_relevant,
                context_total_count=precision.num_total,
            )

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}", exc_info=True)
            return None


@lru_cache
def get_ragas_evaluator() -> RagasEvaluator:
    return RagasEvaluator()

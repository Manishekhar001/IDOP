import asyncio
from functools import lru_cache
from typing import Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.opik import track
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocEvalScore(BaseModel):
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score in [0.0, 1.0]. 1.0 = chunk alone answers the question fully.",
    )
    reason: str = Field(..., description="Short justification for the score.")


_DOC_EVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for a RAG system.\n"
            "You will be given ONE retrieved document chunk and a user question.\n"
            "Return a relevance score in [0.0, 1.0].\n\n"
            "Scoring guide:\n"
            "  1.0 — chunk alone is sufficient to fully answer the question\n"
            "  0.7 — chunk contains strong, directly relevant information\n"
            "  0.5 — chunk is partially relevant (related topic, incomplete answer)\n"
            "  0.3 — chunk is marginally relevant (same domain, no direct answer)\n"
            "  0.0 — chunk is completely irrelevant\n\n"
            "Be conservative with high scores. Also return a short reason.\n"
            "Output JSON only.",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
)

CRAGVerdict = Literal["CORRECT", "AMBIGUOUS", "INCORRECT"]


class CRAGEvaluator:
    def __init__(self) -> None:
        settings = get_settings()
        self._upper_th = settings.crag_upper_threshold
        self._lower_th = settings.crag_lower_threshold

        llm = ChatOpenAI(
            model=settings.memory_llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
        )
        self._eval_chain = _DOC_EVAL_PROMPT | llm.with_structured_output(DocEvalScore)

        logger.info(
            f"CRAGEvaluator ready — "
            f"upper_th={self._upper_th}, lower_th={self._lower_th}"
        )

    @track(name="crag_evaluator_evaluate")
    async def evaluate(
        self, question: str, docs: list[Document]
    ) -> tuple[CRAGVerdict, str, list[Document]]:
        if not docs:
            logger.warning("CRAGEvaluator.evaluate called with empty docs list")
            return "INCORRECT", "No documents retrieved", []

        async def _score_doc(doc: Document) -> tuple[float, Document]:
            try:
                result: DocEvalScore = await self._eval_chain.ainvoke(
                    {"question": question, "chunk": doc.page_content}
                )
                logger.debug(f"Chunk scored {result.score:.2f} — {result.reason[:80]}")
                return result.score, doc
            except Exception as e:
                logger.error(f"Doc eval failed for chunk: {e}")
                return 0.0, doc

        pairs = await asyncio.gather(*[_score_doc(d) for d in docs])

        scores: list[float] = []
        good_docs: list[Document] = []
        for score, doc in pairs:
            scores.append(score)
            if score > self._lower_th:
                good_docs.append(doc)

        if any(s >= self._upper_th for s in scores):
            verdict: CRAGVerdict = "CORRECT"
            reason = (
                f"At least one chunk scored >= {self._upper_th} "
                f"(max={max(scores):.2f})"
            )
        elif all(s < self._lower_th for s in scores):
            verdict = "INCORRECT"
            reason = (
                f"All chunks scored < {self._lower_th} "
                f"(max={max(scores):.2f})"
            )
            good_docs = []
        else:
            verdict = "AMBIGUOUS"
            reason = (
                f"No chunk >= {self._upper_th} but not all < {self._lower_th} "
                f"(max={max(scores):.2f})"
            )

        logger.info(f"CRAG verdict: {verdict} — {reason}")
        return verdict, reason, good_docs


@lru_cache
def get_crag_evaluator() -> CRAGEvaluator:
    return CRAGEvaluator()

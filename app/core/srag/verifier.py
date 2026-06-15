from functools import lru_cache
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.core.llm_factory import get_memory_llm
from app.opik import track
from app.utils.logger import get_logger

logger = get_logger(__name__)

SupportVerdict = Literal["fully_supported", "partially_supported", "no_support"]


class SupportDecision(BaseModel):
    verdict: SupportVerdict = Field(
        ...,
        description=(
            "fully_supported — every claim in the answer is backed by the context.\n"
            "partially_supported — some claims are backed, others are not.\n"
            "no_support — the answer contradicts or ignores the context."
        ),
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Short direct quotes from the context that support the answer.",
    )


_SUPPORT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict factual grounding auditor.\n"
            "Given a question, a context, and an answer, decide whether the answer "
            "is supported by the context.\n\n"
            "Verdicts:\n"
            "  fully_supported    — every claim in the answer is traceable to the context.\n"
            "  partially_supported — some claims are supported, others are not.\n"
            "  no_support          — the answer makes claims not found in or contradicted "
            "by the context.\n\n"
            "Also extract short evidence quotes (max 20 words each) from the context that "
            "back the answer. If no support exists, return an empty list.\n"
            "Output JSON only.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"),
    ]
)

UsefulnessVerdict = Literal["useful", "not_useful"]


class UsefulnessDecision(BaseModel):
    verdict: UsefulnessVerdict = Field(
        ...,
        description=(
            "useful     — the answer clearly and directly addresses the question.\n"
            "not_useful — the answer is vague, off-topic, or incomplete."
        ),
    )
    reason: str = Field(..., description="One-sentence justification.")


_USEFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a response quality judge.\n"
            "Given a question and an answer, decide whether the answer is useful "
            "to the person who asked.\n\n"
            "Verdicts:\n"
            "  useful     — the answer clearly and directly addresses the question.\n"
            "  not_useful — the answer is vague, off-topic, generic, or incomplete.\n\n"
            "Also give a one-sentence reason. Output JSON only.",
        ),
        ("human", "Question: {question}\n\nAnswer:\n{answer}"),
    ]
)

_REVISE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a factual editor.\n"
            "The answer below has claims that are NOT fully supported by the context.\n"
            "Rewrite the answer so that EVERY claim is directly traceable to the context.\n"
            "Remove or qualify any claim that cannot be backed by the context.\n"
            "Do not add new information. Keep the answer concise and helpful.",
        ),
        (
            "human",
            "Question: {question}\n\nContext:\n{context}\n\n"
            "Original answer (needs revision):\n{answer}",
        ),
    ]
)


class SRAGVerifier:
    def __init__(self) -> None:
        llm = get_memory_llm()
        self._support_chain = _SUPPORT_PROMPT | llm.with_structured_output(
            SupportDecision
        )
        self._usefulness_chain = _USEFULNESS_PROMPT | llm.with_structured_output(
            UsefulnessDecision
        )
        self._revise_chain = _REVISE_PROMPT | llm
        logger.info("SRAGVerifier ready")

    @track(name="srag_verify_support")
    async def verify_support(
        self, question: str, context: str, answer: str
    ) -> tuple[SupportVerdict, list[str]]:
        logger.debug("Verifying answer support...")
        try:
            result: SupportDecision = await self._support_chain.ainvoke(
                {"question": question, "context": context, "answer": answer}
            )
            logger.info(
                f"SRAG support: {result.verdict} "
                f"({len(result.evidence)} evidence items)"
            )
            return result.verdict, result.evidence
        except Exception as e:
            logger.error(f"Support verification failed: {e}")
            return "fully_supported", []

    @track(name="srag_verify_usefulness")
    async def verify_usefulness(
        self, question: str, answer: str
    ) -> tuple[UsefulnessVerdict, str]:
        logger.debug("Verifying answer usefulness...")
        try:
            result: UsefulnessDecision = await self._usefulness_chain.ainvoke(
                {"question": question, "answer": answer}
            )
            logger.info(f"SRAG usefulness: {result.verdict} — {result.reason}")
            return result.verdict, result.reason
        except Exception as e:
            logger.error(f"Usefulness verification failed: {e}")
            return "useful", "Verification error — accepting answer as-is."

    @track(name="srag_revise_answer")
    async def revise_answer(self, question: str, context: str, answer: str) -> str:
        logger.info("Revising answer to improve factual grounding...")
        try:
            result = await self._revise_chain.ainvoke(
                {"question": question, "context": context, "answer": answer}
            )
            revised = result.content if hasattr(result, "content") else str(result)
            logger.info("Answer revised successfully")
            return revised
        except Exception as e:
            logger.error(f"Answer revision failed: {e}")
            return answer


@lru_cache
def get_srag_verifier() -> SRAGVerifier:
    return SRAGVerifier()

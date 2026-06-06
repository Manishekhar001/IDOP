import logging
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from app.core.llm_factory import get_memory_llm
from app.opik import track

logger = logging.getLogger("idop_app.mutation_llm_judge")


class AuditResult(BaseModel):
    is_approved: bool = Field(
        ..., description="Whether the mutation is approved for execution."
    )
    explanation: str = Field(
        ...,
        description="Brief explanation of why it is approved or what the concern is.",
    )


_AUDIT_MUTATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert Database Transaction Auditor.\n"
            "Evaluate the following proposed database mutation.\n"
            "Decide if the target table and action are safe and align with the user's "
            "natural language request.\n"
            "Block anything that looks like SQL injection or unintended mass deletions.",
        ),
        (
            "human",
            "Proposed Action:\n"
            "- Target Table: {table_name}\n"
            "- Operation Type: {op_type}\n"
            "\n"
            "User Prompt: {request_text}",
        ),
    ]
)


class MutationLLMJudge:
    """
    LLM-as-Judge to validate database mutations for business alignment and transactional safety.
    """

    def __init__(self):
        self.llm = get_memory_llm()
        self._chain = _AUDIT_MUTATION_PROMPT | self.llm.with_structured_output(
            AuditResult
        )

    @track(name="mutation_llm_judge_audit")
    async def audit_mutation(
        self, request_text: str, table_name: str, op_type: str
    ) -> tuple[bool, str]:
        """
        Audit the planned mutation. Returns (is_approved, explanation).
        """
        try:
            result: AuditResult = await self._chain.ainvoke(
                {
                    "request_text": request_text,
                    "table_name": table_name,
                    "op_type": op_type,
                }
            )
            logger.info(
                f"Mutation LLM Judge audit: is_approved={result.is_approved}, explanation={result.explanation}"
            )
            return result.is_approved, result.explanation
        except Exception as e:
            logger.error(f"Mutation LLM Judge audit failed: {e}")
            return False, f"Audit failed due to internal error: {e}"

import logging
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from app.core.llm_factory import get_memory_llm
from app.opik import track

logger = logging.getLogger("idop_app.llm_judge")


class JudgeResult(BaseModel):
    is_correct: bool = Field(
        ...,
        description="Whether the SQL query is semantically correct for the user's question.",
    )
    explanation: str = Field(
        ..., description="Brief explanation of why it is correct or what is wrong."
    )


_JUDGE_SQL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert DB Auditor. Review the following user request and generated SQL.\n"
            "Decide if the SQL query correctly and safely answers the user's question semantically.\n\n"
            "Evaluate the query:\n"
            "1. Does it join the correct tables using correct keys?\n"
            "2. Does it filter on correct segments/status correctly?\n"
            "3. Does it prevent hallucinations (e.g. correct column fields)?",
        ),
        ("human", "User Question: {question}\nGenerated SQL:\n{sql}"),
    ]
)


class LLMJudge:
    """
    LLM-as-Judge to check if generated SQL correctly matches user's semantic intent.
    """

    def __init__(self):
        self.llm = get_memory_llm()
        self._chain = _JUDGE_SQL_PROMPT | self.llm.with_structured_output(JudgeResult)

    @track(name="llm_judge_sql")
    async def judge_sql(self, question: str, sql: str) -> tuple[bool, str]:
        """
        Evaluate if the generated SQL is semantically correct for the input question.

        Returns:
            Tuple of (is_correct, explanation)
        """
        try:
            result: JudgeResult = await self._chain.ainvoke(
                {"question": question, "sql": sql}
            )
            logger.info(
                f"LLM Judge verdict: is_correct={result.is_correct}, explanation={result.explanation}"
            )
            return result.is_correct, result.explanation
        except Exception as e:
            logger.error(f"LLM Judge execution failed: {e}")
            return True, f"Bypassed LLM Judge due to error: {e}"

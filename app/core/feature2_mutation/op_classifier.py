import logging
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from app.core.llm_factory import get_memory_llm
from app.opik import track

logger = logging.getLogger("idop_app.op_classifier")


class OpVerdict(BaseModel):
    operation: str = Field(
        ..., description="Classified operation. Must be INSERT, UPDATE, or DELETE."
    )


_OP_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a database access query analyzer.\n"
            "Analyze the user's intent to determine if they want to insert new rows, "
            "update existing rows, or delete rows from the database.\n"
            "Respond with exactly one word: INSERT, UPDATE, or DELETE.",
        ),
        ("human", "User Request: {request_text}"),
    ]
)


class OpClassifier:
    """
    Classifies mutation intent (INSERT/UPDATE/DELETE) from a natural language request or structure.
    """

    def __init__(self):
        self.llm = get_memory_llm()
        self._chain = _OP_CLASSIFY_PROMPT | self.llm.with_structured_output(OpVerdict)

    @track(name="op_classifier_classify")
    async def classify_operation(self, request_text: str) -> str:
        """
        Classify operational intent: INSERT, UPDATE, or DELETE.
        """
        try:
            result: OpVerdict = await self._chain.ainvoke(
                {"request_text": request_text}
            )
            verdict = result.operation.strip().upper()
            if verdict in ["INSERT", "UPDATE", "DELETE"]:
                logger.info(f"Classified mutation request as: {verdict}")
                return verdict
            return "INSERT"  # Default fallback
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return "INSERT"

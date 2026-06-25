from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.core.llm_factory import get_chat_llm
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RouteDecision(BaseModel):
    query_type: str = Field(
        ...,
        description="Classified query type. Must be SQL, MUTATION, RAG, CHAT, or HYBRID.",
    )
    reason: str = Field(..., description="Short explanation of the classification.")


_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an intelligent routing agent for IDOP (Intelligent Data Operations Platform).\n"
            "Analyze the following user query and classify it into one of 5 paths:\n\n"
            "1. SQL: Fetching, querying, summarizing, or analyzing transactional data from "
            "database tables (products, customers, orders).\n"
            "2. MUTATION: Performing modifications like inserting, updating, loading or "
            "deleting rows from tables, especially triggered by spreadsheets.\n"
            "3. RAG: Searching for knowledge, policies, guidelines, or non-tabular "
            "instructions in PDFs or TXT documents.\n"
            "4. CHAT: General conversational phrases, greetings, help questions, system "
            "status, or other miscellaneous queries.\n"
            "5. HYBRID: Queries that require BOTH querying transactional database tables "
            "and searching for knowledge/guidelines in documents.\n"
            "\n"
            "Respond with a JSON object containing `query_type` (one of: SQL, MUTATION, "
            "RAG, CHAT, HYBRID) and `reason` (short explanation).",
        ),
        ("human", "User Query: {question}"),
    ]
)


class QueryRouter:
    """
    A 5-class semantic LLM router that classifies questions into SQL, MUTATION, RAG, CHAT, or HYBRID.
    """

    def __init__(self):
        self.llm = get_chat_llm()
        self._chain = _ROUTER_PROMPT | self.llm.with_structured_output(RouteDecision)

    async def route_query(self, question: str) -> str:
        try:
            decision: RouteDecision = await self._chain.ainvoke({"question": question})
            verdict = decision.query_type.upper()
            if verdict in ["SQL", "MUTATION", "RAG", "CHAT", "HYBRID"]:
                logger.info(
                    f"Router classified question as {verdict}. Reason: {decision.reason}"
                )
                return verdict
            return "CHAT"
        except Exception as e:
            logger.error(f"Router execution failed: {e}. Falling back to CHAT.")
            return "CHAT"

import logging
from openai import OpenAI
from pydantic import BaseModel, Field
from app.config import get_settings

logger = logging.getLogger("idop_app.router")


class RouteDecision(BaseModel):
    query_type: str = Field(
        ...,
        description="Classified query type. Must be SQL, MUTATION, RAG, CHAT, or HYBRID."
    )
    reason: str = Field(..., description="Short explanation of the classification.")


class QueryRouter:
    """
    A 5-class semantic LLM router that classifies questions into SQL, MUTATION, RAG, CHAT, or HYBRID.
    """

    def __init__(self):
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.memory_llm_model

    def route_query(self, question: str) -> str:
        prompt = f"""
You are an intelligent routing agent for IDOP (Intelligent Data Operations Platform).
Analyze the following user query and classify it into one of 5 paths:

1. SQL: Fetching, querying, summarizing, or analyzing transactional data from database tables (products, customers, orders).
   Example: "Which products have never been ordered?", "How many SMB customers are in Canada?", "Show total revenue".
2. MUTATION: Performing modifications like inserting, updating, loading or deleting rows from tables, especially triggered by spreadsheets or specific commands.
   Example: "Insert new products from this file", "Update customer segments", "Delete cancelled orders".
3. RAG: Searching for knowledge, policies, guidelines, or non-tabular instructions in PDFs or TXT documents.
   Example: "What is our refund policy?", "Summarize the employee leave rules PDF".
4. CHAT: General conversational phrases, greetings, help questions, system status, or other miscellaneous queries.
   Example: "Hello", "How does this platform work?", "Are systems healthy?".
5. HYBRID: Queries that require BOTH querying transactional database tables (e.g. products, customers, orders) and searching for knowledge, guidelines, policies, or explanations in PDFs/TXT documents at the same time.
   Example: "Get sales data for customer X and compare it against the sales strategy in our PDF guidelines.", "List all products in stock and verify their compliance with our pricing policy document."

User Query: "{question}"

Respond strictly in the following JSON format:
{{
  "query_type": "SQL" / "MUTATION" / "RAG" / "CHAT" / "HYBRID",
  "reason": "Why this classification was chosen"
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            import json
            result_dict = json.loads(response.choices[0].message.content)
            decision = RouteDecision.model_validate(result_dict)
            verdict = decision.query_type.upper()
            if verdict in ["SQL", "MUTATION", "RAG", "CHAT", "HYBRID"]:
                logger.info(f"Router classified question as {verdict}. Reason: {decision.reason}")
                return verdict
            return "CHAT"
        except Exception as e:
            logger.error(f"Router execution failed: {e}. Falling back to CHAT.")
            return "CHAT"

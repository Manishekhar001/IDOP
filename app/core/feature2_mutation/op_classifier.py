import logging
from openai import OpenAI
from app.config import get_settings

logger = logging.getLogger("idop_app.op_classifier")


class OpClassifier:
    """
    Classifies mutation intent (INSERT/UPDATE/DELETE) from a natural language request or structure.
    """

    def __init__(self):
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.memory_llm_model

    def classify_operation(self, request_text: str) -> str:
        """
        Classify operational intent: INSERT, UPDATE, or DELETE.
        """
        prompt = f"""
You are a database access query analyzer.
Analyze the user's intent to determine if they want to insert new rows, update existing rows, or delete rows from the database.

User Request: "{request_text}"

Respond with exactly one word: INSERT, UPDATE, or DELETE.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            verdict = response.choices[0].message.content.strip().upper()
            if verdict in ["INSERT", "UPDATE", "DELETE"]:
                logger.info(f"Classified mutation request as: {verdict}")
                return verdict
            return "INSERT"  # Default fallback
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return "INSERT"

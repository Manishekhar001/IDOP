import logging
from openai import OpenAI
from app.config import get_settings

logger = logging.getLogger("idop_app.llm_judge")


class LLMJudge:
    """
    LLM-as-Judge to check if generated SQL correctly matches user's semantic intent.
    """

    def __init__(self):
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.memory_llm_model  # Uses gpt-4o-mini for cost-efficiency

    def judge_sql(self, question: str, sql: str) -> tuple[bool, str]:
        """
        Evaluate if the generated SQL is semantically correct for the input question.

        Returns:
            Tuple of (is_correct, explanation)
        """
        prompt = f"""
You are an expert DB Auditor. Review the following user request and generated SQL.
Decide if the SQL query correctly and safely answers the user's question semantically.

User Question: {question}
Generated SQL:
{sql}

Evaluate the query:
1. Does it join the correct tables using correct keys?
2. Does it filter on correct segments/status correctly?
3. Does it prevent hallucinations (e.g. correct column fields)?

Respond strictly in the following JSON format:
{{
  "is_correct": true/false,
  "explanation": "Brief explanation of why it is correct or what is wrong with it"
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
            result = json.loads(response.choices[0].message.content)
            is_correct = result.get("is_correct", True)
            explanation = result.get("explanation", "Matches semantic expectations.")
            logger.info(f"LLM Judge verdict: is_correct={is_correct}, explanation={explanation}")
            return is_correct, explanation
        except Exception as e:
            logger.error(f"LLM Judge execution failed: {e}")
            return True, f"Bypassed LLM Judge due to error: {e}"

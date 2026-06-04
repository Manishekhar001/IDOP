import logging
import json
from openai import OpenAI
from app.opik import track
from app.config import get_settings

logger = logging.getLogger("idop_app.mutation_llm_judge")


class MutationLLMJudge:
    """
    LLM-as-Judge to validate database mutations for business alignment and transactional safety.
    """

    def __init__(self):
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.memory_llm_model

    @track(name="mutation_llm_judge_audit")
    def audit_mutation(
        self, request_text: str, table_name: str, op_type: str
    ) -> tuple[bool, str]:
        """
        Audit the planned mutation. Returns (is_approved, explanation).
        """
        prompt = f"""
You are an expert Database Transaction Auditor.
Evaluate the following proposed database mutation.

Proposed Action:
- Target Table: {table_name}
- Operation Type: {op_type}

User Prompt Triggering Action: "{request_text}"

Decide if the target table and action are safe and align with the user's natural language request.
Block anything that looks like SQL injection or unintended mass deletions.

Respond strictly in the following JSON format:
{{
  "is_approved": true/false,
  "explanation": "Brief explanation of why it is approved or what the concern is"
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            is_approved = result.get("is_approved", True)
            explanation = result.get(
                "explanation", "Matches business transaction alignment."
            )
            logger.info(
                f"Mutation LLM Judge audit: is_approved={is_approved}, explanation={explanation}"
            )
            return is_approved, explanation
        except Exception as e:
            logger.error(f"Mutation LLM Judge audit failed: {e}")
            return False, f"Audit failed due to internal error: {e}"

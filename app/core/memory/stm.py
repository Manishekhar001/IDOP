from functools import lru_cache
import uuid

from langchain_core.messages import HumanMessage, RemoveMessage

from app.opik import track
from app.config import get_settings
from app.core.llm_factory import get_memory_llm
from app.utils.logger import get_logger

logger = get_logger(__name__)


class STMSummarizer:
    def __init__(self) -> None:
        settings = get_settings()
        self._threshold = settings.stm_message_threshold
        self._llm = get_memory_llm()
        logger.info(f"STMSummarizer ready — threshold={self._threshold} messages")

    def should_summarize(self, messages: list) -> bool:
        return len(messages) > self._threshold

    @track(name="stm_summarize")
    async def summarize(
        self, messages: list, existing_summary: str
    ) -> tuple[str, list]:
        if existing_summary:
            prompt_text = (
                f"Existing summary:\n{existing_summary}\n\n"
                "Extend the summary to include the new conversation above. "
                "Be concise."
            )
        else:
            prompt_text = (
                "Summarise the conversation above concisely. "
                "Capture key facts, user preferences, and conclusions."
            )

        messages_for_summary = list(messages) + [
            HumanMessage(content=prompt_text, id=str(uuid.uuid4()))
        ]

        logger.info(
            f"Summarising conversation — {len(messages)} messages, "
            f"existing_summary={'yes' if existing_summary else 'no'}"
        )

        response = await self._llm.ainvoke(messages_for_summary)
        new_summary: str = response.content

        messages_to_delete = messages[:-2]
        remove_ops = [RemoveMessage(id=m.id) for m in messages_to_delete if m.id]

        logger.info(
            f"Summary generated — deleted {len(remove_ops)} messages, "
            f"kept {len(messages) - len(remove_ops)}"
        )
        return new_summary, remove_ops


@lru_cache
def get_stm_summarizer() -> STMSummarizer:
    return STMSummarizer()

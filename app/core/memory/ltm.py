import uuid
from functools import lru_cache

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.opik import track
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryItem(BaseModel):
    text: str = Field(..., description="Short atomic fact about the user.")
    is_new: bool = Field(
        ...,
        description="True if this fact is NEW vs existing memories, False if duplicate.",
    )


class MemoryDecision(BaseModel):
    should_write: bool = Field(
        ...,
        description="True if there is any memory-worthy information in the message.",
    )
    memories: list[MemoryItem] = Field(default_factory=list)


_MEMORY_PROMPT = """\
You are responsible for maintaining accurate long-term user memory.

CURRENT USER DETAILS (existing memories):
{existing_memories}

TASK:
- Review the user's latest message.
- Extract user-specific information worth storing long-term:
    identity, stable preferences, ongoing projects, goals, tools used.
- For each item set is_new=true ONLY if it adds genuinely NEW information
  compared to CURRENT USER DETAILS.
- If it duplicates existing memory, set is_new=false.
- Keep each memory as one short atomic sentence.
- No speculation — only facts explicitly stated by the user.
- If nothing is memory-worthy, return should_write=false and an empty list.
"""


class LTMService:
    def __init__(self) -> None:
        settings = get_settings()
        self._llm = ChatOpenAI(
            model=settings.memory_llm_model,
            temperature=settings.memory_llm_temperature,
            api_key=settings.openai_api_key,
        )
        self._extractor = self._llm.with_structured_output(MemoryDecision)
        logger.info(f"LTMService ready — model={settings.memory_llm_model}")

    @staticmethod
    def _namespace(user_id: str) -> tuple:
        return ("user", user_id, "details")

    @track(name="ltm_read_memories")
    async def read_memories(self, store, user_id: str) -> str:
        ns = self._namespace(user_id)
        items = await store.asearch(ns)
        if not items:
            logger.debug(f"LTM: no memories found for user={user_id}")
            return "(empty)"
        memories = "\n".join(it.value.get("data", "") for it in items)
        logger.debug(f"LTM: read {len(items)} memories for user={user_id}")
        return memories

    @track(name="ltm_extract_and_store")
    async def extract_and_store(self, store, user_id: str, user_message: str) -> int:
        existing = await self.read_memories(store, user_id)
        ns = self._namespace(user_id)

        try:
            decision: MemoryDecision = await self._extractor.ainvoke(
                [
                    SystemMessage(
                        content=_MEMORY_PROMPT.format(existing_memories=existing)
                    ),
                    {"role": "user", "content": user_message},
                ]
            )
        except Exception as e:
            logger.error(f"LTM extraction failed: {e}")
            return 0

        written = 0
        if decision.should_write:
            for mem in decision.memories:
                if mem.is_new and mem.text.strip():
                    await store.aput(ns, str(uuid.uuid4()), {"data": mem.text.strip()})
                    written += 1
                    logger.debug(f"LTM stored: '{mem.text.strip()}'")

        logger.info(
            f"LTM extraction done — {written} new facts stored for user={user_id}"
        )
        return written


@lru_cache
def get_ltm_service() -> LTMService:
    return LTMService()

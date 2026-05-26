import logging
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from app.config import get_settings

logger = logging.getLogger("idop_app.hyde")


class HydeHypotheses(BaseModel):
    """Structured hypothetical document excerpts for query expansion."""
    hypotheses: list[str] = Field(
        ...,
        description="A list of 2-3 sentence hypothetical document passages that directly answer the query."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "hypotheses": [
                        "The refund policy states that customers can return products within 30 days for a full refund.",
                        "For delivered orders, refunds are processed back to the original payment method within 5 business days."
                    ]
                }
            ]
        }
    }


_HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at generating hypothetical document passages that could answer a given question.\n"
            "Generate exactly {num_hypotheses} diverse, detailed passages that represent different ways the question could be answered in actual documents.\n"
            "Each passage should be 2-3 sentences long and read like an excerpt from a real document (e.g., policy guide, system manual, database documentation).\n"
            "Output JSON matching the schema.",
        ),
        ("human", "Question: {query}"),
    ]
)


class HydeService:
    """
    Hypothetical Document Embeddings (HyDE) service using structured Pydantic outputs.
    """

    def __init__(self):
        settings = get_settings()
        llm = ChatOpenAI(
            model=settings.memory_llm_model,  # gpt-4o-mini
            temperature=0.7,
            api_key=settings.openai_api_key,
        )
        self._hyde_chain = _HYDE_PROMPT | llm.with_structured_output(HydeHypotheses)

    async def generate_hypothetical_documents_async(self, query: str, num_hypotheses: int = 3) -> list[str]:
        """
        Generate hypothetical answers to improve retrieval (async).
        """
        logger.info(f"HyDE: Generating {num_hypotheses} hypothetical documents for: '{query[:50]}...'")
        try:
            result: HydeHypotheses = await self._hyde_chain.ainvoke(
                {"query": query, "num_hypotheses": num_hypotheses}
            )
            if result and result.hypotheses:
                logger.info(f"HyDE: Generated {len(result.hypotheses)} hypothetical documents successfully.")
                return result.hypotheses
            return [query]
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}. Falling back to original query.")
            return [query]

    def generate_hypothetical_documents(self, query: str, num_hypotheses: int = 3) -> list[str]:
        """
        Generate hypothetical answers to improve retrieval (sync wrapper).
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If running inside an existing event loop, run as threadsafe future to avoid blockages
                future = asyncio.run_coroutine_threadsafe(
                    self.generate_hypothetical_documents_async(query, num_hypotheses), loop
                )
                return future.result()
            else:
                return loop.run_until_complete(
                    self.generate_hypothetical_documents_async(query, num_hypotheses)
                )
        except Exception as e:
            logger.warning(f"HyDE sync execution exception: {e}. Retrying with new event loop...")
            try:
                new_loop = asyncio.new_event_loop()
                res = new_loop.run_until_complete(
                    self.generate_hypothetical_documents_async(query, num_hypotheses)
                )
                new_loop.close()
                return res
            except Exception as e2:
                logger.error(f"HyDE sync execution completely failed: {e2}")
                return [query]

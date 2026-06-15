import asyncio
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.llm_factory import get_memory_llm
from app.opik import track
from app.utils.logger import get_logger

logger = get_logger(__name__)


class WebQuery(BaseModel):
    query: str = Field(..., description="Focused web-search query (6-14 keywords).")


_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a focused web-search query.\n"
            "Rules:\n"
            "  - Keep it short: 6-14 keywords.\n"
            "  - If the question implies recency (recent/latest/last week), "
            "add a time constraint such as '(last 30 days)'.\n"
            "  - Do NOT answer the question.\n"
            "  - Return JSON with a single key: query.",
        ),
        ("human", "Question: {question}"),
    ]
)


class WebSearchService:
    def __init__(self) -> None:
        settings = get_settings()

        llm = get_memory_llm()
        self._rewrite_chain = _REWRITE_PROMPT | llm.with_structured_output(WebQuery)

        self._tavily = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=settings.tavily_max_results,
        )
        logger.info(
            f"WebSearchService ready — tavily max_results={settings.tavily_max_results}"
        )

    @track(name="web_search_rewrite")
    async def rewrite_query(self, question: str) -> str:
        logger.debug(f"Rewriting query for: {question[:80]}")
        result: WebQuery = await self._rewrite_chain.ainvoke({"question": question})
        logger.info(f"Rewritten web query: {result.query}")
        return result.query

    @track(name="web_search_execute")
    async def search(self, query: str) -> list[Document]:
        logger.info(f"Tavily web search: '{query}'")
        try:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                None, lambda: self._tavily.invoke({"query": query})
            )
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []

        web_docs: list[Document] = []
        for r in results:
            if isinstance(r, dict):
                title = r.get("title", "")
                url = r.get("url", "")
                raw_content = r.get("content", "") or r.get("snippet", "")
            elif isinstance(r, str):
                title = ""
                url = r
                raw_content = r
            else:
                logger.warning(f"Unexpected Tavily result type: {type(r)}")
                continue
            text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{raw_content}"
            web_docs.append(
                Document(
                    page_content=text,
                    metadata={"title": title, "url": url, "source": url},
                )
            )

        logger.info(f"Tavily returned {len(web_docs)} results")
        return web_docs


@lru_cache
def get_web_search_service() -> WebSearchService:
    return WebSearchService()

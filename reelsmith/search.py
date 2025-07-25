import asyncio
import httpx

from bs4 import BeautifulSoup
from langchain_core.tools import tool

from reelsmith.llm import LLM
from reelsmith.stub import State


class Search:
    def __init__(self, llm: LLM, instruction: str) -> None:
        self.llm = llm
        self.instruction = instruction

    @staticmethod
    async def _extract_content(url: str):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=5)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")
                return ' '.join(p.get_text() for p in soup.find_all("p")).strip().replace('\n', ' ')

        except httpx.HTTPStatusError:
            return ""

    async def _summarize_helper(self, url: str):
        content = await self._extract_content(url)

        if not content:
            return ""

        prompt = f"Summarize this article in 200 words:\n\n{content}"

        try:
            result = await self.llm.ainvoke(prompt)
            return result.content.strip()

        except:
            return ""

    @tool
    async def summarize(self, state: State) -> State:
        urls = self.search(state.topic)

        tasks = [self._summarize_helper(url) for url in urls]
        summaries = await asyncio.gather(*tasks)

        return State(search_summary=summaries, **state.model_dump())

    def search(self, topic: str, max_results: int = 5) -> list[str]:
        pass


class SearXNGSearch(Search):
    def __init__(self, llm: LLM, instruction: str, searxng_url="http://localhost:8888") -> None:
        super().__init__(llm, instruction)
        self.searxng_url = searxng_url

    def search(self, topic: str, max_results: int = 5):
        try:
            response = httpx.get(f"{self.searxng_url}/search", params={
                "q": topic,
                "format": "json"
            })
            results = response.json().get("results", [])
            urls = [r.get("url") for r in results if r.get("url")]
            return urls[:max_results]

        except Exception:
            return []

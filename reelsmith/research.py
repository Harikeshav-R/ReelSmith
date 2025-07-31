import threading

import httpx

from queue import Queue

from bs4 import BeautifulSoup

from reelsmith.llm import LLM
from reelsmith.stub import State


class Research:
    def __init__(self, llm: LLM, instruction: str) -> None:
        self.llm = llm
        self.instruction = instruction

    @staticmethod
    def _extract_content(url: str) -> str:
        try:
            response = httpx.get(url, timeout=5)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            return ' '.join(p.get_text() for p in soup.find_all("p")).strip().replace('\n', ' ')

        except httpx.HTTPStatusError:
            return ""

    def _summarize(self, url: str, output_queue: Queue):
        content = self._extract_content(url)

        if not content:
            output_queue.put("")
            return

        prompt = f"Summarize this article in 500 words:\n\n{content}"

        try:
            result = self.llm.invoke(prompt)
            output_queue.put(result.content.strip())

        except Exception:
            output_queue.put("")

    def research(self, state: State) -> State:
        urls = self._search(state.topic)
        output_queue = Queue()
        threads = []

        for url in urls:
            thread = threading.Thread(target=self._summarize, args=(url, output_queue))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        summaries = [output_queue.get() for _ in urls]

        state.search_summary = summaries
        return state

    def _search(self, topic: str, max_results: int = 5) -> list[str]:
        pass


class SearXNGResearch(Research):
    def __init__(self, llm: LLM, instruction: str, searxng_url="http://localhost:8888") -> None:
        super().__init__(llm, instruction)
        self.searxng_url = searxng_url

    def _search(self, topic: str, max_results: int = 3):
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

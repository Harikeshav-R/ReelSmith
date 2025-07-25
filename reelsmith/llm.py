from typing import Sequence, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableConfig
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_ollama.chat_models import ChatOllama


class LLM:
    def __init__(self, model: str, reasoning: bool = False) -> None:
        self.model: str = model
        self.reasoning: bool = reasoning

        self.llm: BaseChatModel | None = None

    def invoke(self,
               input: PromptValue | str | Sequence[BaseMessage | list[str] | tuple[str, str] | str | dict[str, Any]],
               config: RunnableConfig | None = None,
               *,
               stop: list[str] | None = None,
               **kwargs: Any) -> BaseMessage:
        if self.llm is None:
            raise ValueError("LLM not initialized")

        return self.llm.invoke(input=input, config=config, stop=stop, **kwargs)

    async def ainvoke(self,
                      input: PromptValue | str | Sequence[
                          BaseMessage | list[str] | tuple[str, str] | str | dict[str, Any]],
                      config: RunnableConfig | None = None,
                      *,
                      stop: list[str] | None = None,
                      **kwargs: Any) -> BaseMessage:
        if self.llm is None:
            raise ValueError("LLM not initialized")

        return await self.llm.ainvoke(input=input, config=config, stop=stop, **kwargs)


class OllamaLLM(LLM):
    def __init__(self, model: str, reasoning: bool = False):
        super().__init__(model, reasoning)
        self.llm = ChatOllama(model=model, reasoning=reasoning)


class GoogleLLM(LLM):
    def __init__(self, model: str, reasoning: bool = False):
        super().__init__(model, reasoning)
        self.llm = ChatGoogleGenerativeAI(model=model)

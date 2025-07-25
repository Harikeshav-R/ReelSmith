from langchain_core.tools import tool

from reelsmith.stub import ScriptState
from reelsmith.llm import LLM


class ScriptGenerator:
    def __init__(self, llm: LLM, instruction: str):
        self.llm = llm
        self.instruction = instruction

    @tool
    async def generate(self, script_state: ScriptState) -> ScriptState:
        """
        Generate a script from a given topic and research performed on that topic by surfing the web.
        """
        prompt = (
            f"You are a video scriptwriter.\n"
            f"Topic: {script_state.topic}\n"
            f"Use the following research:\n{script_state.search_summary}\n\n"
            f"Do not use markdown or titles. Do not annotate anything. Output only the script.\n"
            f"{self.instruction}"
        )

        response = self.llm.invoke(prompt)

        return ScriptState(
            topic=script_state.topic,
            search_summary=script_state.search_summary,
            script=response.content.strip(),
            audio_path=None,
            caption_path=None,
            video_path=None,
        )

from reelsmith.stub import State
from reelsmith.llm import LLM


class ScriptGenerator:
    def __init__(self, llm: LLM, instruction: str):
        self.llm = llm
        self.instruction = instruction

    def generate(self, state: State) -> State:
        """
        Generate a script from a given topic and research performed on that topic by surfing the web.
        """
        prompt = (
            f"You are a video scriptwriter.\n"
            f"Topic: {state.topic}\n"
            f"Use the following research:\n{state.search_summary}\n\n"
            f"Do not use markdown or titles. Do not annotate anything. Output only the script.\n"
            f"{self.instruction}"
        )

        response = self.llm.invoke(prompt)
        state.script = response.content.strip()

        return state

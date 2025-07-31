import asyncio

from langgraph.graph import StateGraph, END

from reelsmith.stub import State
from reelsmith.llm import GoogleLLM, OllamaLLM
from reelsmith.research import SearXNGResearch
from reelsmith.script import ScriptGenerator


def input_node(state: State) -> State:
    return state.model_dump()


def research_node(state: State) -> State:
    search_engine = SearXNGResearch(GoogleLLM("gemini-2.5-flash"), "")
    # search_engine = SearXNGResearch(OllamaLLM("mistral"), "")
    state = asyncio.run(search_engine.research(state)).model_dump()
    return state


def script_node(state: State) -> State:
    script_generator = ScriptGenerator(GoogleLLM("gemini-2.5-flash"), "")

    state = script_generator.generate_script_words(state)

    script_generator = ScriptGenerator(OllamaLLM("mistral"), "")

    state = script_generator.generate_image_prompts(state)

    return state.model_dump()


builder = StateGraph(State)
builder.add_node("input", input_node)
builder.add_node("research", research_node)
builder.add_node("script", script_node)

builder.set_entry_point("input")
builder.add_edge("input", "research")
builder.add_edge("research", "script")
builder.add_edge("script", END)

graph = builder.compile()

if __name__ == "__main__":
    topic = input("Enter a video topic: ")
    result = State(**graph.invoke(State(topic=topic)))

    print("\n--- Generated Script ---\n")
    print(result)

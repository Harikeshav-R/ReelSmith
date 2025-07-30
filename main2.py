import asyncio

from langgraph.graph import StateGraph, END

from reelsmith.stub import State
from reelsmith.llm import OllamaLLM
from reelsmith.research import SearXNGResearch
from reelsmith.script import ScriptGenerator


def input_node(state: State) -> State:
    return state.model_dump()


def research_node(state: State) -> State:
    search_engine = SearXNGResearch(OllamaLLM("huihui_ai/qwen3-abliterated:8b"), "Summarize this article in 200 words:")
    return asyncio.run(search_engine.research(state)).model_dump()


def script_node(state: State) -> State:
    script_generator = ScriptGenerator(OllamaLLM("huihui_ai/qwen3-abliterated:8b"), "You are a video scriptwriter.")
    return script_generator.generate(state).model_dump()


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
    print(result.script)

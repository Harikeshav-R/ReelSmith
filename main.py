import concurrent.futures
import subprocess
import os
import shutil

from typing import Optional

import requests
import soundfile as sf
import torch
import whisperx

from bs4 import BeautifulSoup
from kokoro import KPipeline
from langchain_ollama.chat_models import ChatOllama
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

LLM_MODEL = "huihui_ai/qwen3-abliterated:8b"


# ---------- STATE ----------

class ScriptState(BaseModel):
    topic: str
    search_summary: Optional[str] = None
    script: Optional[str] = None
    audio_path: Optional[str] = None
    caption_path: Optional[str] = None
    video_path: Optional[str] = None


# ---------- SEARCH & SUMMARY ----------

def search_searxng(query, searxng_url="http://localhost:8888", max_results=5):
    try:
        response = requests.get(f"{searxng_url}/search", params={
            "q": query,
            "format": "json"
        })
        results = response.json().get("results", [])[:max_results]
        urls = [r.get("url") for r in results if r.get("url")]
        return urls
    except Exception:
        return []


def extract_content(url):
    try:
        html = requests.get(url, timeout=5).text
        soup = BeautifulSoup(html, "html.parser")
        return ' '.join(p.get_text() for p in soup.find_all("p")).strip().replace('\n', ' ')
    except:
        return ""


def summarize_content_parallel(urls):
    llm = ChatOllama(model=LLM_MODEL, reasoning=False)

    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    def summarize(url):
        content = extract_content(url)
        if not content:
            return ""
        prompt = f"Summarize this article in 3-4 sentences:\n\n{content}"
        try:
            return llm.invoke(prompt).content.strip()
        except:
            return ""

    with concurrent.futures.ThreadPoolExecutor() as executor:
        summaries = list(executor.map(summarize, urls))
    return "\n\n".join([s for s in summaries if s])


# ---------- SCRIPT ----------

def generate_script(state: ScriptState) -> dict:
    llm = ChatOllama(model=LLM_MODEL, reasoning=False)
    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    prompt = (
        f"You are a video scriptwriter.\n"
        f"Topic: {state.topic}\n"
        f"Use the following research:\n{state.search_summary}\n\n"
        f"Write a clear, engaging, informative narrator script in 3–4 paragraphs. "
        f"Do not use markdown or titles. Do not annotate anything. Output only the script."
    )
    response = llm.invoke(prompt)
    return {**state.model_dump(), "script": response.content}


# ---------- TTS ----------

def synthesize_audio(state: ScriptState) -> dict:
    pipeline = KPipeline(lang_code="a")
    generator = pipeline(state.script, voice='af_bella', speed=1.25)

    for i, (gs, ps, audio) in enumerate(generator):
        sf.write(f'output/{i}.wav', audio, 24000)

    merge_audio_files("output")

    return {**state.model_dump(), "audio_path": "output/narration.wav"}


def merge_audio_files(output_dir):
    file_list_path = os.path.join(output_dir, 'files.txt')

    wav_files = sorted([f for f in os.listdir(output_dir)
                        if f.endswith('.wav')])

    with open(file_list_path, 'w') as f:
        for wav_file in wav_files:
            f.write(f"file '{wav_file}'\n")

    merged_output = f'narration.wav'

    ffmpeg_cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'files.txt',
        '-c', 'copy',
        merged_output
    ]

    print("\nMerging audio files...")
    subprocess.run(ffmpeg_cmd, cwd=output_dir)
    print(f"Merged audio saved as: {output_dir}/{merged_output}")


# ---------- CAPTIONS ----------

def chunk_words(words, chunk_size=3):
    chunks = []
    for i in range(0, len(words), chunk_size):
        w = words[i:i + chunk_size]
        try:
            text = " ".join(word.get("word", word.get("text", "")).strip() for word in w)
            start = w[0]["start"]
            end = w[-1]["end"]
            chunks.append((start, end, text))
        except KeyError:
            continue
    return chunks


def format_time(s):
    ms = int((s - int(s)) * 1000)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h:02}:{m:02}:{sec:02},{ms:03}"


def write_srt(subs, path="output/narration.srt"):
    with open(path, "w") as f:
        for i, (start, end, text) in enumerate(subs, 1):
            f.write(f"{i}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{text.strip()}\n\n")


def generate_captions(state: ScriptState) -> dict:
    model = whisperx.load_model("large-v2", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")
    result = model.transcribe(state.audio_path)

    align_model, metadata = whisperx.load_align_model(language_code="en", device="cpu")
    aligned = whisperx.align(result["segments"], align_model, metadata, state.audio_path, device="cpu")
    words = aligned["word_segments"]

    subs = chunk_words(words, chunk_size=3)
    write_srt(subs)
    return {**state.model_dump(), "caption_path": "output/narration.srt"}


def embed_audio_and_subtitles(state: ScriptState) -> dict:
    original_video = "video.mp4"
    audio_path = state.audio_path
    srt_path = state.caption_path
    output_path = "output/final_video.mp4"

    try:
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", original_video,
            "-i", audio_path,
            "-vf", f"subtitles={srt_path}:force_style='FontName=Arial,FontSize=28'",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-shortest",
            output_path
        ], check=True)
        print(f"Final video with audio and subtitles saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print("FFmpeg failed:", e)

    return {**state.model_dump(), "video_path": output_path}


# ------------- CLEAN UP --------------
def clean_up():
    shutil.copyfile("output/final_video.mp4", "final_video.mp4")
    shutil.rmtree("output")


# ---------- LANGGRAPH NODES ----------

def input_node(state: ScriptState) -> dict:
    return state.model_dump()


def web_search_node(state: ScriptState) -> dict:
    urls = search_searxng(state.topic)
    summary = summarize_content_parallel(urls)
    return {**state.model_dump(), "search_summary": summary}


# ---------- GRAPH BUILD ----------

builder = StateGraph(ScriptState)
builder.add_node("input", input_node)
builder.add_node("search", web_search_node)
builder.add_node("write_script", generate_script)
builder.add_node("tts", synthesize_audio)
builder.add_node("captions", generate_captions)
builder.add_node("embed_audio_and_subtitles", embed_audio_and_subtitles)

builder.set_entry_point("input")
builder.add_edge("input", "search")
builder.add_edge("search", "write_script")
builder.add_edge("write_script", "tts")
builder.add_edge("tts", "captions")
builder.add_edge("captions", "embed_audio_and_subtitles")
builder.add_edge("embed_audio_and_subtitles", END)

graph = builder.compile()

# ---------- RUN ----------

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    topic = input("Enter a video topic: ")
    result = ScriptState(**graph.invoke(ScriptState(topic=topic)))

    print("\n--- Generated Script ---\n")
    print(result.script)
    print("\nSaved audio to:", result.audio_path)
    print("Saved captions to:", result.caption_path)

    clean_up()

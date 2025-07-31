import tempfile
from pathlib import Path

import torch
import whisperx

from reelsmith.stub import State


class SubtitleGenerator:
    def __init__(self, model: str = "large-v2", device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 compute_type="int8") -> None:
        self.model = model
        self.device = device
        self.compute_type = compute_type

    @staticmethod
    def _chunk_words(words, chunk_size=3) -> list[tuple[int, int, str]]:
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

    @staticmethod
    def format_time(seconds: int) -> str:
        ms = int((seconds - int(seconds)) * 1000)
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        sec = int(seconds % 60)
        return f"{h:02}:{m:02}:{sec:02},{ms:03}"

    @staticmethod
    def write_srt(subtitles: list[tuple[int, int, str]], srt_file_path: Path) -> None:
        with open(srt_file_path, "w") as f:
            for i, (start, end, text) in enumerate(subtitles, 1):
                f.write(f"{i}\n")
                f.write(f"{SubtitleGenerator.format_time(start)} --> {SubtitleGenerator.format_time(end)}\n")
                f.write(f"{text.strip()}\n\n")

    def generate_captions(self, state: State) -> State:
        model = whisperx.load_model(self.model, device=self.device, compute_type=self.compute_type)
        audio = whisperx.load_audio(str(state.final_audio_path))
        result = model.transcribe(audio)

        align_model, metadata = whisperx.load_align_model(language_code="en", device=self.device)
        aligned = whisperx.align(result["segments"], align_model, metadata, str(state.final_audio_path),
                                 device=self.device)
        words = aligned["word_segments"]

        captions_file = tempfile.NamedTemporaryFile(delete=False, suffix=".srt", mode="w")

        subs = self._chunk_words(words, chunk_size=3)
        self.write_srt(subs, srt_file_path=Path(captions_file.name))

        state.caption_path = Path(captions_file.name)

        return state

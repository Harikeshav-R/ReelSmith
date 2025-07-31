import tempfile
import wave

import ffmpeg
import soundfile as sf

from pathlib import Path

from kokoro import KPipeline

from reelsmith.stub import State


class TTSGenerator:
    def __init__(self, voice: str, speed: float = 1.0):
        self.voice = voice
        self.speed = speed

    @staticmethod
    def _concatenate_wav_files(input_files: list[Path], output_file: Path) -> None:
        for file in input_files:
            if not file.exists():
                raise FileNotFoundError(f"{file} does not exist")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
            for file in input_files:
                tf.write(f"file '{file.as_posix()}'\n")
            concat_list_path = Path(tf.name)

        try:
            (
                ffmpeg
                .input(str(concat_list_path), format='concat', safe=0)
                .output(str(output_file), acodec='copy')
                .run(overwrite_output=True, quiet=True)
            )
        finally:
            concat_list_path.unlink(missing_ok=True)

    @staticmethod
    def _generate_sentences(state: State) -> list[str]:
        sentences = []

        for i in state.image_segments.image_segments:
            words = state.script.script_words[i.word_range[0]:i.word_range[1] + 1]
            sentence = " ".join(words)
            sentences.append(sentence)

        return sentences

    def _generate_audio(self, text: str) -> tuple[float, Path]:
        pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
        generator = pipeline(text, voice=self.voice, speed=self.speed)

        generated_audio_files = []
        final_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb")

        for gs, ps, audio in generator:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb")
            temp_path = Path(temp_file.name)
            generated_audio_files.append(temp_path)
            temp_file.close()

            sf.write(temp_path, audio, 24000)

        self._concatenate_wav_files(generated_audio_files, output_file=Path(final_audio_file.name))

        for temp_path in generated_audio_files:
            temp_path.unlink(missing_ok=True)

        with wave.open(final_audio_file.name, 'r') as wav_file:
            audio_duration = wav_file.getnframes() / wav_file.getframerate()

        return audio_duration, Path(final_audio_file.name)

    def run_tts(self, state: State) -> State:
        sentences = self._generate_sentences(state)

        audio_durations = []
        audio_files = []
        final_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb")

        for sentence in sentences:
            audio_duration, audio_path = self._generate_audio(sentence)
            audio_durations.append(audio_duration)
            audio_files.append(audio_path)

        self._concatenate_wav_files(audio_files,
                                    output_file=Path(final_audio_file.name))

        for audio_file in audio_files:
            audio_file.unlink(missing_ok=True)

        state.final_audio_path = Path(final_audio_file.name)
        state.audio_clip_durations = audio_durations
        return state

from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, Field


class ImagePromptSegment(BaseModel):
    prompt: str = Field(..., description="A prompt to generate an image relevant to a section of the script.")
    word_range: Tuple[int, int] = Field(...,
                                        description="A tuple (start_index, end_index) indicating the range of word indices the image relates to.")


class ImageSegmentList(BaseModel):
    image_segments: list[ImagePromptSegment]


class Script(BaseModel):
    script_plaintext: str
    script_words: list[str]


class State(BaseModel):
    topic: str = Field(description="The topic for the video script.")
    search_summary: Optional[list[str]] = Field(
        description="The summary of the research performed by searching the web.", default=None)
    script: Optional[Script] = Field(description="The video script in plain text and word split form.", default=None)
    image_segments: Optional[ImageSegmentList] = Field(description="The image prompts for each range of words",
                                                       default=None)
    audio_path: Optional[Path] = Field(description="The path to the generated audio.", default=None)
    caption_path: Optional[Path] = Field(description="The path to the generated captions.", default=None)
    video_path: Optional[Path] = Field(description="The path to the final generated video.", default=None)

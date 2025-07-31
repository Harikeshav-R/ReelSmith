from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class State(BaseModel):
    topic: str = Field(description="The topic for the video script.")
    search_summary: Optional[list[str]] = Field(
        description="The summary of the research performed by searching the web.", default=None)
    script: Optional[str] = Field(description="The generated script.", default=None)
    audio_path: Optional[Path] = Field(description="The path to the generated audio.", default=None)
    caption_path: Optional[Path] = Field(description="The path to the generated captions.", default=None)
    video_path: Optional[Path] = Field(description="The path to the final generated video.", default=None)


class ImagePromptSegment(BaseModel):
    prompt: str = Field(..., description="A prompt to generate an image relevant to a section of the script.")
    word_range: Tuple[int, int] = Field(...,
                                        description="A tuple (start_index, end_index) indicating the range of word indices the image relates to.")


class VideoScript(BaseModel):
    script_words: List[str] = Field(..., description="List of words forming the complete script.")
    image_segments: List[ImagePromptSegment] = Field(...,
                                                     description="List of image prompts and their corresponding word ranges.")

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class State(BaseModel):
    topic: str = Field(description="The topic for the video script.")
    search_summary: Optional[list[str]] = Field(
        description="The summary of the research performed by searching the web.")
    script: Optional[str] = Field(description="The generated script.")
    audio_path: Optional[Path] = Field(description="The path to the generated audio.")
    caption_path: Optional[Path] = Field(description="The path to the generated captions.")
    video_path: Optional[Path] = Field(description="The path to the final generated video.")

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class State(BaseModel):
    topic: str = Field(description="The topic for the video script.")
    search_summary: Optional[list[str]] = Field(
        description="The summary of the research performed by searching the web.", default=None)
    script: Optional[str] = Field(description="The generated script.", default=None)
    audio_path: Optional[Path] = Field(description="The path to the generated audio.", default=None)
    caption_path: Optional[Path] = Field(description="The path to the generated captions.", default=None)
    video_path: Optional[Path] = Field(description="The path to the final generated video.", default=None)

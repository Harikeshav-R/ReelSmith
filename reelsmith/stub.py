from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class ScriptState(BaseModel):
    topic: str
    search_summary: Optional[list[str]] = None
    script: Optional[str] = None
    audio_path: Optional[Path] = None
    caption_path: Optional[Path] = None
    video_path: Optional[Path] = None

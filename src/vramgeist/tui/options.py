from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TUIOptions:
    high_contrast: bool = False
    theme: str | None = None
    json_export: Path | None = None
    log_level: str = "WARNING"
    start_in_preview: bool = False
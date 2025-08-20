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
    optimize_for: str = "throughput"  # "throughput" or "memory"
    debug: bool = False
    # Validation settings
    validate_settings: bool = False
    llama_bin: str | None = None
    validation_timeout: float = 30.0

    def __post_init__(self):
        """Initialize validation settings from saved config if available"""
        # Only auto-load if not explicitly set
        if self.llama_bin is None:
            try:
                from ..config_persist import get_llama_bin_path, get_validation_timeout, should_enable_validation_by_default
                self.llama_bin = get_llama_bin_path()
                self.validation_timeout = get_validation_timeout()
                # Enable validation by default if we have a working llama.cpp path
                if not self.validate_settings and should_enable_validation_by_default():
                    self.validate_settings = True
            except ImportError:
                pass  # Config persistence not available
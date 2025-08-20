from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from .options import TUIOptions


@dataclass(slots=True)
class TUIState:
    files: List[Path]
    options: TUIOptions = field(default_factory=TUIOptions)
    results_by_path: Dict[Path, Dict[str, Any]] = field(default_factory=dict)
    errors_by_path: Dict[Path, str] = field(default_factory=dict)
    selected_index: int = 0

    def add_result(self, path: Path, result: Dict[str, Any]) -> None:
        self.results_by_path[path] = result

    def add_error(self, path: Path, message: str) -> None:
        self.errors_by_path[path] = message

    def toggle_optimization_mode(self) -> None:
        """Toggle between throughput and memory optimization"""
        self.options.optimize_for = "memory" if self.options.optimize_for == "throughput" else "throughput"

    def toggle_debug(self) -> None:
        """Toggle debug mode on/off"""
        self.options.debug = not self.options.debug

    def toggle_validation(self) -> None:
        """Toggle validation on/off (only if llama_bin is available)"""
        if self.options.llama_bin:
            self.options.validate_settings = not self.options.validate_settings

    def can_validate(self) -> bool:
        """Check if validation is possible (llama_bin is configured)"""
        return self.options.llama_bin is not None

    def select_next(self) -> None:
        if not self.files:
            return
        self.selected_index = min(self.selected_index + 1, len(self.files) - 1)

    def select_prev(self) -> None:
        if not self.files:
            return
        self.selected_index = max(self.selected_index - 1, 0)

    def get_status_line(self, path: Path) -> str:
        name = path.name
        if path in self.errors_by_path:
            return f"× {name}: {self.errors_by_path[path]}"
        if path in self.results_by_path:
            res = self.results_by_path[path]
            
            # Extract basic info
            model = res.get("model_name") or (res.get("model", {}).get("name") if isinstance(res.get("model"), dict) else "unknown")
            rec = res.get("recommendation", {})
            ctx = rec.get("max_context", "?")
            gpu = rec.get("gpu_layers", "?")
            
            # Add validation indicator
            validation_icon = ""
            if res.get("validation"):
                validation_info = res["validation"]
                if validation_info.get("validated"):
                    validation_icon = "🔒"  # Verified/validated
                else:
                    validation_icon = "⚠️"   # Validation failed
            
            return f"✓ {name}{validation_icon}: ctx={ctx}, gpu={gpu}"
        return f"⏳ Analyzing: {name}"
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class TUIState:
    files: List[Path]
    results_by_path: Dict[Path, Dict[str, Any]] = field(default_factory=dict)
    errors_by_path: Dict[Path, str] = field(default_factory=dict)
    selected_index: int = 0

    def add_result(self, path: Path, result: Dict[str, Any]) -> None:
        self.results_by_path[path] = result

    def add_error(self, path: Path, message: str) -> None:
        self.errors_by_path[path] = message

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
            model = res.get("model_name") or res.get("model") or "unknown"
            ctx = res.get("max_context") or res.get("context_length") or "?"
            gpu = res.get("recommended_gpu_layers") or res.get("gpu_layers") or "?"
            return f"✓ {name}: model={model}, ctx={ctx}, gpu_layers={gpu}"
        return f"⏳ Analyzing: {name}"
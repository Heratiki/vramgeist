from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..state import TUIState

# Lazy import textual components inside functions/classes to avoid hard dependency at import time.
# This module defines MainScreen with minimal rendering logic.


class MainScreen:
    def __init__(self, state: "TUIState") -> None:
        self.state = state

    def compose(self):
        from textual.widgets import Header, Footer, Static
        from textual.containers import Vertical

        yield Header(show_clock=False, tall=False, name="vramgeist (experimental TUI)")
        body = Vertical(id="body")
        # Initial population; Textual will re-render when we update content
        for line in self._lines():
            body.mount(Static(line, expand=False))
        yield body
        yield Footer()

    def on_mount(self) -> None:
        # Nothing complex for now; updates happen when state changes via controller.
        pass

    def refresh_list(self) -> None:
        # Rebuild the list of lines from state to keep it deterministic and simple.
        from textual.widgets import Static
        from textual.app import App

        app = App.app  # type: ignore[attr-defined]
        body = app.query_one("#body")  # type: ignore[assignment]
        # Clear and repopulate
        body.remove_children()
        for line in self._lines():
            body.mount(Static(line, expand=False))

    def _lines(self) -> Iterable[str]:
        for path in self.state.files:
            yield self.state.get_status_line(path)
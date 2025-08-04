from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, List, Dict, Any

from .options import TUIOptions
from .state import TUIState

# NOTE: Keep imports of textual/rich lazy and inside functions/classes that need them
# so that importing this module doesn't hard-require extras unless run_tui is actually invoked.


def _lazy_textual_imports():
    # Import and return textual parts on demand
    from textual.app import App, ComposeResult
    from textual.widgets import Header, Footer, Static
    from textual.containers import Vertical
    from textual import events
    return App, ComposeResult, Header, Footer, Static, Vertical, events


class VramgeistAppBase:
    """A minimal controller that uses Textual via lazy imports.
    We keep this wrapper to isolate textual usage and simplify testing when extras are missing.
    """

    def __init__(
        self,
        paths: List[Path],
        options: TUIOptions,
        analyze_fn: Callable[[Path], Dict[str, Any]],
        state: TUIState | None = None,
        max_workers: int = 4,
    ) -> None:
        self.paths = paths
        self.options = options
        self.analyze_fn = analyze_fn
        self.state = state or TUIState(files=paths)
        self.max_workers = max_workers

    def _make_textual_app_class(self):
        App, ComposeResult, Header, Footer, Static, Vertical, events = _lazy_textual_imports()
        # import screen after textual is available

        state = self.state
        analyze_fn = self.analyze_fn
        max_workers = self.max_workers

        class VramgeistApp(App):
            def _reload_files(self, directory: Path | None = None) -> None:
                # Helper to reload files and directories, respecting show_hidden, and add '..' entry
                dir_path = directory or getattr(self._state, "current_dir", None) or Path.cwd()
                if not hasattr(self._state, "show_hidden"):
                    self._state.show_hidden = False
                entries = []
                # Add parent directory entry unless at root
                if dir_path != dir_path.parent:
                    entries.append("..")
                # List directories first, then files
                for p in sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                    if not self._state.show_hidden and p.name.startswith("."):
                        continue
                    entries.append(p)
                self._state.files = entries
                if hasattr(self._state, "current_dir"):
                    self._state.current_dir = dir_path
                self._refresh_list()
            CSS = """
            #body {
                overflow: auto;
            }
            """

            TITLE = "vramgeist (experimental TUI)"

            def __init__(self) -> None:
                super().__init__()
                self._state = state
                self._executor: ThreadPoolExecutor | None = None
                self._tasks: list[asyncio.Task[None]] = []

            def compose(self) -> ComposeResult:
                yield Header(show_clock=False, name="vramgeist (experimental TUI)")
                with Vertical(id="body") as body:
                    for line in self._lines():
                        yield Static(line, expand=False)
                self._body = body  # type: ignore[attr-defined]
                yield Footer()

            def _lines(self) -> Iterable[str]:
                for entry in self._state.files:
                    if entry == "..":
                        yield "[..]"
                    else:
                        yield self._state.get_status_line(entry)

            async def on_mount(self) -> None:
                # start background analysis
                self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vg-analyze")
                await self._spawn_analysis_tasks()

            async def _spawn_analysis_tasks(self) -> None:
                # limit concurrency using executor; submit per file and gather
                loop = asyncio.get_running_loop()
                for path in self._state.files:
                    task = asyncio.create_task(self._analyze_one(loop, path))
                    self._tasks.append(task)

                # As tasks complete, update UI
                for task in asyncio.as_completed(self._tasks):
                    try:
                        await task
                    except Exception:
                        # already recorded error; continue
                        pass
                    self._refresh_list()

            async def _analyze_one(self, loop: asyncio.AbstractEventLoop, path: Path) -> None:
                try:
                    # Run CPU-bound or blocking IO in a thread to avoid blocking event loop
                    result = await loop.run_in_executor(self._executor, analyze_fn, path)
                    self._state.add_result(path, result)
                except Exception as e:
                    self._state.add_error(path, str(e))

            def _refresh_list(self) -> None:
                from textual.widgets import Static
                # Clear and repopulate
                self._body.remove_children()
                for line in self._lines():
                    self._body.mount(Static(line, expand=False))

            async def on_key(self, event) -> None:
                # Navigation: j/k/up/down
                if event.key in ("j", "down"):
                    self._state.select_next()
                    self._refresh_list()
                elif event.key in ("k", "up"):
                    self._state.select_prev()
                    self._refresh_list()
                # Quit: q/escape
                elif event.key in ("q", "escape"):
                    await self.action_quit()
                # Toggle hidden files: '.' or 'period'
                elif event.key in (".", "period"):
                    if not hasattr(self._state, "show_hidden"):
                        self._state.show_hidden = False
                    self._state.show_hidden = not self._state.show_hidden
                    dir_path = getattr(self._state, "current_dir", None) or Path.cwd()
                    self._reload_files(dir_path)
                # Process: F2/Space
                elif event.key in ("f2", "space"):
                    selected_entry = self._state.get_selected_path() if hasattr(self._state, "get_selected_path") else None
                    if selected_entry and selected_entry != "..":
                        loop = asyncio.get_running_loop()
                        await self._analyze_one(loop, selected_entry)
                        self._refresh_list()
                # Parent directory: Backspace/Left
                elif event.key in ("backspace", "left"):
                    dir_path = getattr(self._state, "current_dir", None) or Path.cwd()
                    parent_dir = dir_path.parent if dir_path != dir_path.parent else dir_path
                    if hasattr(self._state, "current_dir"):
                        self._state.current_dir = parent_dir
                    self._reload_files(parent_dir)
                # Home directory: Home
                elif event.key == "home":
                    home_dir = Path.home()
                    if hasattr(self._state, "current_dir"):
                        self._state.current_dir = home_dir
                    self._reload_files(home_dir)
                # Enter: navigate into directory or parent, process file
                elif event.key in ("enter",):
                    selected_entry = self._state.get_selected_path() if hasattr(self._state, "get_selected_path") else None
                    dir_path = getattr(self._state, "current_dir", None) or Path.cwd()
                    if selected_entry == "..":
                        parent_dir = dir_path.parent if dir_path != dir_path.parent else dir_path
                        self._state.current_dir = parent_dir
                        self._reload_files(parent_dir)
                    elif isinstance(selected_entry, Path) and selected_entry.is_dir():
                        self._state.current_dir = selected_entry
                        self._reload_files(selected_entry)
                    elif isinstance(selected_entry, Path) and selected_entry.is_file():
                        loop = asyncio.get_running_loop()
                        await self._analyze_one(loop, selected_entry)
                        self._refresh_list()

            async def on_unmount(self) -> None:
                # Clean up executor
                if self._executor is not None:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                    self._executor = None

        return VramgeistApp

    async def run_async(self) -> int:
        VramgeistApp = self._make_textual_app_class()
        app = VramgeistApp()
        await app.run_async()
        return 0


def run_tui(paths: List[Path], options: TUIOptions) -> int:
    """Entry point used by CLI to run the experimental TUI.

    This must NOT print to stdout. It may raise KeyboardInterrupt which the CLI should map to exit 130.
    """
    # If no paths provided, use Textual-based file browser mode
    if not paths:
        from .file_browser import browse_files
        
        # Use the Textual file browser to select files/directories
        selected = browse_files(start_dir=Path.cwd(), select_files=True, select_dirs=True)
        
        if selected is None:
            # User cancelled
            return 130
        
        # Convert selected path to list of GGUF files
        selected_path = Path(selected)
        if selected_path.is_file() and selected_path.suffix.lower() == ".gguf":
            paths = [selected_path]
        elif selected_path.is_dir():
            # Find all GGUF files in directory
            gguf_files = list(selected_path.glob("*.gguf"))
            if not gguf_files:
                import sys
                sys.stderr.write(f"No GGUF files found in directory: {selected_path}\n")
                sys.stderr.flush()
                return 1
            paths = gguf_files
        else:
            import sys
            sys.stderr.write(f"Selected path is not a GGUF file or directory: {selected_path}\n")
            sys.stderr.flush()
            return 1

    # Proceed with analysis using the TUI
    # Lazy import of ui for analysis function to avoid early imports
    from .. import ui as ui_module  # legacy ui remains unchanged

    def analyze_gguf_file(path: Path) -> Dict[str, Any]:
        # Call existing analyze function and return its result as dict
        return ui_module.analyze_gguf_file(path)

    controller = VramgeistAppBase(paths=paths, options=options, analyze_fn=analyze_gguf_file)
    # Run with asyncio; let textual manage event loop via run_async
    return asyncio.run(controller.run_async())
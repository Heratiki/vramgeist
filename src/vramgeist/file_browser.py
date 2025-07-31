from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import os


class PromptToolkitMissingError(RuntimeError):
    pass


def _require_prompt_toolkit() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import HSplit, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.styles import Style
        from prompt_toolkit.keys import Keys  # noqa: F401
        return Application, KeyBindings, Layout, HSplit, Window, FormattedTextControl, Style
    except Exception as exc:  # pragma: no cover - import guard
        raise PromptToolkitMissingError(
            "Interactive file browser requires 'prompt_toolkit'. "
            "Install with: pip install 'prompt_toolkit>=3.0,<4.0'"
        ) from exc


def _list_dir(path: Path, show_hidden: bool) -> list[tuple[str, Path, bool]]:
    try:
        entries = list(path.iterdir())
    except PermissionError:
        entries = []
    dirs: list[tuple[str, Path, bool]] = []
    files: list[tuple[str, Path, bool]] = []
    for p in entries:
        name = p.name
        if not show_hidden and name.startswith("."):
            continue
        if p.is_dir():
            dirs.append((name + os.sep, p, True))
        else:
            files.append((name, p, False))
    dirs.sort(key=lambda t: t[0].lower())
    files.sort(key=lambda t: t[0].lower())
    # Add parent directory '..' entry unless at root
    if path.parent != path:
        dirs = [(".." + os.sep, path.parent, True)] + dirs
    return dirs + files


def _render_title(cwd: Path, show_hidden: bool) -> str:
    hidden = "[hidden: on]" if show_hidden else "[hidden: off]"
    # Breadcrumbs for current path
    breadcrumbs = f"Browsing: {cwd}" if cwd != Path.home() else f"Browsing: ~"
    return (
        f"{breadcrumbs}  {hidden}\n"
        "[↑/↓] Move  [Enter] Select/Open  [Backspace/←] Up Dir  [Home] Home  [.] Toggle Hidden  [q/ESC] Cancel"
    )


def browse(
    start_dir: str | os.PathLike[str] | None = None,
    select_files: bool = True,
    select_dirs: bool = True,
) -> Optional[str]:
    """
    Interactive file/folder browser using prompt_toolkit.

    Returns absolute path of selected entry or None if cancelled.
    """
    Application, KeyBindings, Layout, HSplit, Window, FormattedTextControl, Style = _require_prompt_toolkit()

    cwd = Path(start_dir) if start_dir is not None else Path.cwd()
    cwd = cwd.expanduser().resolve()

    show_hidden = False
    selection_index = 0
    entries: list[tuple[str, Path, bool]] = _list_dir(cwd, show_hidden)
    selected_path: Optional[Path] = None

    # State for metadata panel
    show_metadata = False
    metadata_lines = []

    def refresh_entries() -> None:
        nonlocal entries, selection_index, show_metadata, metadata_lines
        entries = _list_dir(cwd, show_hidden)
        if not entries:
            selection_index = 0
        else:
            selection_index = max(0, min(selection_index, len(entries) - 1))
        show_metadata = False
        metadata_lines = []

    def get_lines() -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = []
        # Split title and help bar into separate lines
        title = _render_title(cwd, show_hidden)
        for tline in title.split("\n"):
            lines.append(("class:title", tline))
        if not entries:
            lines.append(("class:empty", "<empty directory>"))
            return lines
        for idx, (name, _p, is_dir) in enumerate(entries):
            # Parent directory '..' entry
            if name == ".." + os.sep:
                prefix = "↩ "
                line = f"{prefix}{name}"
                style = "class:parent"
            else:
                prefix = "📁 " if is_dir else "📄 "
                line = f"{prefix}{name}"
                if idx == selection_index:
                    style = "class:current"
                elif is_dir:
                    style = "class:dir"
                else:
                    style = "class:file"
            lines.append((style, line))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    def _up(event) -> None:
        nonlocal selection_index
        if entries:
            selection_index = (selection_index - 1) % len(entries)

    @kb.add("down")
    def _down(event) -> None:
        nonlocal selection_index
        if entries:
            selection_index = (selection_index + 1) % len(entries)

    @kb.add("enter")
    def _enter(event) -> None:
        nonlocal cwd, show_metadata, metadata_lines
        if not entries:
            return
        entry = entries[selection_index]
        _name, p, is_dir = entry
        if is_dir:
            # Browse into folder
            cwd = p.resolve()
            refresh_entries()
        else:
            # Show metadata panel for file
            show_metadata = True
            metadata_lines = []
            try:
                stat = p.stat()
                metadata_lines.append(f"Name: {p.name}")
                metadata_lines.append(f"Size: {stat.st_size} bytes")
                metadata_lines.append(f"Modified: {stat.st_mtime}")
                metadata_lines.append(f"Created: {stat.st_ctime}")
                metadata_lines.append(f"Type: {p.suffix if p.suffix else 'N/A'}")
            except Exception as e:
                metadata_lines.append(f"Error reading metadata: {e}")

    @kb.add("f2")
    @kb.add(" ")
    def _select(event) -> None:
        nonlocal selected_path, show_metadata, metadata_lines
        if not entries:
            return
        entry = entries[selection_index]
        _name, p, is_dir = entry
        if is_dir and select_dirs:
            selected_path = p.resolve()
            event.app.exit()
        elif not is_dir and select_files:
            selected_path = p.resolve()
            event.app.exit()
        show_metadata = False
        metadata_lines = []

    @kb.add("backspace")
    @kb.add("left")
    def _updir(event) -> None:
        nonlocal cwd
        parent = cwd.parent
        if parent != cwd:
            cwd = parent
            refresh_entries()

    @kb.add("home")
    def _home(event) -> None:
        nonlocal cwd
        home = Path.home()
        if home.exists():
            cwd = home
            refresh_entries()

    @kb.add(".")
    def _toggle_hidden(event) -> None:
        nonlocal show_hidden
        show_hidden = not show_hidden
        refresh_entries()

    @kb.add("q")
    @kb.add("escape")
    def _cancel(event) -> None:
        nonlocal selected_path
        selected_path = None
        event.app.exit()

    from prompt_toolkit.layout.dimension import Dimension

    def get_formatted_text():
        # If metadata panel is active, show it below the browser
        lines = get_lines()
        out = [(style, text + "\n") for style, text in lines]
        if show_metadata and metadata_lines:
            out.append(("class:meta_title", "\n[File Metadata]\n"))
            for mline in metadata_lines:
                out.append(("class:meta", mline + "\n"))
        return out

    body = FormattedTextControl(get_formatted_text, focusable=True)
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.layout.scrollable_pane import ScrollOffsets
    # Set a minimum and maximum height for the menu window
    menu_height = Dimension(min=10, max=20)
    root_container = HSplit([
        Window(
            content=body,
            height=menu_height,
            always_hide_cursor=False,
            wrap_lines=False,
            scroll_offsets=ScrollOffsets(top=2, bottom=2)
        )
    ])
    layout = Layout(root_container)
    style = Style.from_dict(
        {
            "title": "bold",
            "current": "reverse",
            "dir": "bold blue",
            "file": "",
            "parent": "bold magenta",
            "empty": "italic #888888",
            "meta_title": "bold underline",
            "meta": "italic #888888",
        }
    )

    app = Application(layout=layout, key_bindings=kb, full_screen=True, style=style)
    refresh_entries()
    app.run()
    return str(selected_path) if selected_path is not None else None
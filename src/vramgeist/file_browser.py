from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, List

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


def _find_gguf_files(directory: Path) -> List[Path]:
    """Find all GGUF files in a directory recursively"""
    gguf_files = []
    try:
        for item in directory.rglob("*.gguf"):
            if item.is_file():
                gguf_files.append(item)
    except PermissionError:
        pass
    return sorted(gguf_files)


def _process_file_with_vramgeist(filepath: Path) -> List[str]:
    """Process a single GGUF file and return detailed analysis results"""
    try:
        # Import VRAMGeist modules
        from .ui import analyze_gguf_file
        from .config import DEFAULT_CONFIG
        
        # Run full analysis
        analysis = analyze_gguf_file(str(filepath), DEFAULT_CONFIG)
        
        lines = []
        lines.append(f"=== VRAMGEIST Analysis: {filepath.name} ===")
        lines.append("")
        
        # Hardware info
        lines.append("Hardware Detection:")
        lines.append(f"  GPU VRAM: {analysis.get('gpu_vram_mb', 'Unknown')} MB")
        lines.append(f"  System RAM: {analysis.get('total_ram_gb', 'Unknown')} GB")
        lines.append("")
        
        # Model info
        model_size = analysis.get('model_size_mb', 'Unknown')
        model_size_str = f"{model_size:.1f} MB" if isinstance(model_size, (int, float)) else str(model_size)
        lines.append("Model Information:")
        lines.append(f"  Size: {model_size_str}")
        lines.append(f"  Layers: {analysis.get('layer_count', 'Unknown')}")
        lines.append("")
        
        # Analysis results
        if 'analysis_results' in analysis:
            results = analysis['analysis_results']
            lines.append("Optimal Configuration:")
            lines.append(f"  GPU Layers: {results.get('best_gpu_layers', 'Unknown')}")
            
            max_context = results.get('max_context', 'Unknown')
            max_context_str = f"{max_context:,}" if isinstance(max_context, (int, float)) else str(max_context)
            lines.append(f"  Max Context: {max_context_str}")
            
            vram_usage = results.get('vram_usage_mb', 'Unknown')
            vram_usage_str = f"{vram_usage:.1f} MB" if isinstance(vram_usage, (int, float)) else str(vram_usage)
            lines.append(f"  VRAM Usage: {vram_usage_str}")
            
            ram_usage = results.get('ram_usage_mb', 'Unknown')
            ram_usage_str = f"{ram_usage:.1f} MB" if isinstance(ram_usage, (int, float)) else str(ram_usage)
            lines.append(f"  RAM Usage: {ram_usage_str}")
            lines.append("")
            
            # Show multiple configurations if available
            if 'configurations' in results:
                lines.append("Available Configurations:")
                for config in results['configurations'][:5]:  # Show top 5
                    gpu_layers = config.get('gpu_layers', 'N/A')
                    max_context = config.get('max_context', 'N/A')
                    vram_usage = config.get('vram_usage', 'N/A')
                    
                    max_context_str = f"{max_context:,}" if isinstance(max_context, (int, float)) else str(max_context)
                    vram_usage_str = f"{vram_usage:.0f}MB" if isinstance(vram_usage, (int, float)) else str(vram_usage)
                    
                    lines.append(f"  Layers: {gpu_layers}, Context: {max_context_str}, VRAM: {vram_usage_str}")
        
        # Warnings
        if 'warnings' in analysis and analysis['warnings']:
            lines.append("")
            lines.append("Warnings:")
            for warning in analysis['warnings']:
                lines.append(f"  ⚠ {warning}")
        
        return lines
        
    except Exception as e:
        return [f"Error analyzing {filepath.name}: {str(e)}"]


def _process_folder_with_vramgeist(directory: Path) -> List[str]:
    """Process all GGUF files in a folder and return summary results"""
    try:
        from .ui import analyze_gguf_file
        from .config import DEFAULT_CONFIG
        
        gguf_files = _find_gguf_files(directory)
        
        if not gguf_files:
            return [f"No GGUF files found in {directory.name}"]
        
        lines = []
        lines.append(f"=== VRAMGEIST Folder Analysis: {directory.name} ===")
        lines.append(f"Found {len(gguf_files)} GGUF file(s)")
        lines.append("")
        
        # Process each file and show summary
        for gguf_file in gguf_files:
            try:
                analysis = analyze_gguf_file(str(gguf_file), DEFAULT_CONFIG)
                
                size_mb = analysis.get('model_size_mb', 0)
                size_str = f"{size_mb:.1f}MB" if isinstance(size_mb, (int, float)) else str(size_mb)
                
                if 'analysis_results' in analysis:
                    results = analysis['analysis_results']
                    gpu_layers = results.get('best_gpu_layers', 'N/A')
                    max_context = results.get('max_context', 'N/A')
                    vram_usage = results.get('vram_usage_mb', 0)
                    
                    max_context_str = f"{max_context:,}" if isinstance(max_context, (int, float)) else str(max_context)
                    vram_usage_str = f"{vram_usage:.0f}MB" if isinstance(vram_usage, (int, float)) else str(vram_usage)
                    
                    lines.append(f"📄 {gguf_file.name}")
                    lines.append(f"   Size: {size_str} | Layers: {gpu_layers} | Context: {max_context_str} | VRAM: {vram_usage_str}")
                else:
                    lines.append(f"📄 {gguf_file.name}")
                    lines.append(f"   Size: {size_str} | Analysis failed")
                    
            except Exception as e:
                lines.append(f"📄 {gguf_file.name}")
                lines.append(f"   Error: {str(e)}")
            
            lines.append("")
        
        return lines
        
    except Exception as e:
        return [f"Error processing folder {directory.name}: {str(e)}"]


def _render_title(cwd: Path, show_hidden: bool) -> str:
    hidden = "[hidden: on]" if show_hidden else "[hidden: off]"
    # Breadcrumbs for current path
    breadcrumbs = f"Browsing: {cwd}" if cwd != Path.home() else f"Browsing: ~"
    return (
        f"{breadcrumbs}  {hidden}\n"
        "[↑/↓] Move  [PgUp/PgDn] Jump  [Enter] Select/Open  [P] Process w/VRAMGeist  [Backspace/←] Up Dir  [Home] Home  [.] Toggle Hidden  [q/ESC] Cancel"
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
    
    # Viewport state for scrolling
    viewport_top = 0
    viewport_height = 15  # Number of visible file entries

    def refresh_entries() -> None:
        nonlocal entries, selection_index, show_metadata, metadata_lines, viewport_top
        entries = _list_dir(cwd, show_hidden)
        if not entries:
            selection_index = 0
        else:
            selection_index = max(0, min(selection_index, len(entries) - 1))
        show_metadata = False
        metadata_lines = []
        viewport_top = 0  # Reset scroll position

    def update_viewport() -> None:
        """Ensure the selected item is visible in the viewport"""
        nonlocal viewport_top
        if not entries:
            return
        
        # If selection is above viewport, scroll up
        if selection_index < viewport_top:
            viewport_top = selection_index
        # If selection is below viewport, scroll down
        elif selection_index >= viewport_top + viewport_height:
            viewport_top = selection_index - viewport_height + 1
        
        # Ensure viewport doesn't go below 0 or beyond entries
        viewport_top = max(0, min(viewport_top, max(0, len(entries) - viewport_height)))

    def get_header_lines() -> list[tuple[str, str]]:
        """Get the fixed header content"""
        lines: list[tuple[str, str]] = []
        # Split title and help bar into separate lines
        title = _render_title(cwd, show_hidden)
        for tline in title.split("\n"):
            lines.append(("class:title", tline))
        
        # Add prominent current directory display
        lines.append(("class:current_dir", f"\n📂 Current Directory: {cwd}\n"))
        return lines

    def get_file_list_lines() -> list[tuple[str, str]]:
        """Get the scrollable file list content"""
        lines: list[tuple[str, str]] = []
        
        if not entries:
            lines.append(("class:empty", "<empty directory>"))
            return lines
        
        # Show only the visible portion of entries
        visible_entries = entries[viewport_top:viewport_top + viewport_height]
        
        for i, (name, _p, is_dir) in enumerate(visible_entries):
            actual_idx = viewport_top + i
            # Parent directory '..' entry
            if name == ".." + os.sep:
                prefix = "↩ "
                line = f"{prefix}{name}"
                style = "class:parent"
            else:
                prefix = "📁 " if is_dir else "📄 "
                line = f"{prefix}{name}"
                if actual_idx == selection_index:
                    style = "class:current"
                elif is_dir:
                    style = "class:dir"
                else:
                    style = "class:file"
            lines.append((style, line))
        
        # Add scroll indicator if there are more entries
        if len(entries) > viewport_height:
            total_entries = len(entries)
            visible_start = viewport_top + 1
            visible_end = min(viewport_top + viewport_height, total_entries)
            scroll_info = f"[{visible_start}-{visible_end} of {total_entries}]"
            lines.append(("class:scroll_info", f"\n{scroll_info}"))
        
        return lines

    kb = KeyBindings()

    @kb.add("up")
    def _up(event) -> None:
        nonlocal selection_index
        if entries:
            selection_index = (selection_index - 1) % len(entries)
            update_viewport()

    @kb.add("down")
    def _down(event) -> None:
        nonlocal selection_index
        if entries:
            selection_index = (selection_index + 1) % len(entries)
            update_viewport()

    @kb.add("pageup")
    def _page_up(event) -> None:
        nonlocal selection_index
        if entries:
            selection_index = max(0, selection_index - 10)
            update_viewport()

    @kb.add("pagedown")
    def _page_down(event) -> None:
        nonlocal selection_index
        if entries:
            selection_index = min(len(entries) - 1, selection_index + 10)
            update_viewport()

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
            update_viewport()
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
            update_viewport()

    @kb.add("home")
    def _home(event) -> None:
        nonlocal cwd
        home = Path.home()
        if home.exists():
            cwd = home
            refresh_entries()
            update_viewport()

    @kb.add(".")
    def _toggle_hidden(event) -> None:
        nonlocal show_hidden
        show_hidden = not show_hidden
        refresh_entries()
        update_viewport()

    @kb.add("p")
    def _process_with_vramgeist(event) -> None:
        """Process selected file/folder with VRAMGeist analysis"""
        nonlocal show_metadata, metadata_lines
        if not entries:
            return
        
        # Get selected entry
        entry = entries[selection_index]
        _name, p, is_dir = entry
        
        # Process the selection
        show_metadata = True
        metadata_lines = []
        
        try:
            if is_dir:
                metadata_lines = _process_folder_with_vramgeist(p)
            else:
                # Check if it's a GGUF file
                if p.suffix.lower() == '.gguf':
                    metadata_lines = _process_file_with_vramgeist(p)
                else:
                    metadata_lines = ["Error: Selected file is not a GGUF model file"]
        except Exception as e:
            metadata_lines = [f"Error processing: {e}"]

    @kb.add("q")
    @kb.add("escape")
    def _cancel(event) -> None:
        nonlocal selected_path
        selected_path = None
        event.app.exit()

    from prompt_toolkit.layout.dimension import Dimension

    def get_header_formatted_text():
        """Get formatted header text (fixed content)"""
        lines = get_header_lines()
        return [(style, text + "\n") for style, text in lines]

    def get_file_list_formatted_text():
        """Get formatted file list text (scrollable content)"""
        lines = get_file_list_lines()
        out = [(style, text + "\n") for style, text in lines]
        
        # Add metadata panel if active
        if show_metadata and metadata_lines:
            out.append(("class:meta_title", "\n[File Metadata]\n"))
            for mline in metadata_lines:
                out.append(("class:meta", mline + "\n"))
        return out

    # Create separate controls for header and file list
    header_control = FormattedTextControl(get_header_formatted_text, focusable=False)
    file_list_control = FormattedTextControl(get_file_list_formatted_text, focusable=True)
    
    # Create layout with fixed header and scrollable file list
    root_container = HSplit([
        Window(
            content=header_control,
            height=Dimension.exact(4),  # Fixed height for header
            wrap_lines=False,
        ),
        Window(
            content=file_list_control,
            always_hide_cursor=False,
            wrap_lines=False,
        )
    ])
    layout = Layout(root_container)
    style = Style.from_dict(
        {
            "title": "bold",
            "current": "reverse",
            "current_dir": "bold cyan underline",
            "dir": "bold blue",
            "file": "",
            "parent": "bold magenta",
            "empty": "italic #888888",
            "meta_title": "bold underline",
            "meta": "italic #888888",
            "scroll_info": "italic #666666",
        }
    )

    app = Application(layout=layout, key_bindings=kb, full_screen=True, style=style)
    refresh_entries()
    app.run()
    return str(selected_path) if selected_path is not None else None
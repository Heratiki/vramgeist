from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Callable, Any

from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Header, Footer, ListView, ListItem, Label, Static
from textual.binding import Binding
from textual.screen import Screen
from textual import events


class FileBrowserItem:
    """Represents a file or directory item in the browser."""
    
    def __init__(self, path: Path, is_parent: bool = False):
        self.path = path
        self.is_parent = is_parent
        self.is_dir = path.is_dir() if not is_parent else True
        
    @property
    def display_name(self) -> str:
        if self.is_parent:
            return "↩ .."
        elif self.is_dir:
            return f"📁 {self.path.name}{os.sep}"
        else:
            return f"📄 {self.path.name}"
    
    @property
    def sort_key(self) -> tuple:
        # Sort parent first, then directories, then files (all case-insensitive)
        if self.is_parent:
            return (0, "")
        elif self.is_dir:
            return (1, self.path.name.lower())
        else:
            return (2, self.path.name.lower())


class FileBrowserWidget(ListView):
    """File browser list widget with keyboard navigation."""
    
    def __init__(self, current_dir: Path, show_hidden: bool = False):
        super().__init__()
        self.current_dir = current_dir.resolve()
        self.show_hidden = show_hidden
        self.selected_path: Optional[Path] = None
        self._items: List[FileBrowserItem] = []
    
    def on_mount(self) -> None:
        """Initialize file listing after widget is mounted."""
        self._refresh_files()
    
    def _refresh_files(self) -> None:
        """Refresh the file listing for current directory."""
        self.clear()
        self._items = []
        
        try:
            # Add parent directory entry (unless at root)
            if self.current_dir.parent != self.current_dir:
                self._items.append(FileBrowserItem(self.current_dir.parent, is_parent=True))
            
            # Add directory contents
            for path in self.current_dir.iterdir():
                # Skip hidden files if not showing them
                if not self.show_hidden and path.name.startswith('.'):
                    continue
                self._items.append(FileBrowserItem(path))
                
        except PermissionError:
            # Handle permission errors gracefully
            pass
        
        # Sort items
        self._items.sort(key=lambda x: x.sort_key)
        
        # Add to ListView if widget is mounted
        if self.is_attached:
            for item in self._items:
                list_item = ListItem(Label(item.display_name))
                list_item.item_data = item  # Store the FileBrowserItem
                self.append(list_item)
    
    def navigate_to(self, path: Path) -> None:
        """Navigate to a new directory."""
        if path.is_dir():
            self.current_dir = path.resolve()
            self._refresh_files()
            self.index = 0  # Reset selection to top
    
    def toggle_hidden(self) -> None:
        """Toggle showing hidden files."""
        self.show_hidden = not self.show_hidden
        current_index = self.index
        self._refresh_files()
        # Try to maintain selection position
        if current_index < len(self._items):
            self.index = current_index
        else:
            self.index = max(0, len(self._items) - 1)
        # Force refresh the display
        self.refresh()
    
    def get_selected_item(self) -> Optional[FileBrowserItem]:
        """Get the currently selected file browser item."""
        if 0 <= self.index < len(self._items):
            return self._items[self.index]
        return None
    
    def select_current(self) -> Optional[Path]:
        """Select the current item and return its path."""
        item = self.get_selected_item()
        if item:
            if item.is_parent:
                return item.path  # Parent directory path
            else:
                return item.path
        return None


class FileBrowserScreen(Screen):
    """Full-screen file browser interface."""
    
    CSS = """
    .hidden {
        display: none;
    }
    
    #file_panel {
        width: 70%;
    }
    
    #metadata_panel {
        width: 30%;
        border-left: solid $primary;
        padding: 1;
    }
    
    #metadata_header {
        text-style: bold;
        color: $primary;
    }
    
    #metadata_content {
        height: 100%;
        overflow-y: auto;
    }
    """
    
    BINDINGS = [
        Binding("up,k", "cursor_up", "Move up"),
        Binding("down,j", "cursor_down", "Move down"),
        Binding("pageup", "page_up", "Page up"),
        Binding("pagedown", "page_down", "Page down"),
        Binding("home", "go_home", "Go home"),
        Binding("backspace,left", "go_up", "Go up directory"),
        Binding("enter", "select_or_enter", "Select/Enter"),
        Binding("f2,space", "select_current", "Select"),
        Binding("period", "toggle_hidden", "Toggle hidden"),
        Binding("q,escape", "cancel_or_hide", "Cancel/Hide"),
    ]
    
    def __init__(self, start_dir: Path = None, select_files: bool = True, select_dirs: bool = True):
        super().__init__()
        self.start_dir = start_dir or Path.cwd()
        self.select_files = select_files
        self.select_dirs = select_dirs
        self.selected_path: Optional[Path] = None
        self.cancelled = False
        self.show_metadata = False
        self.metadata_file: Optional[Path] = None
    
    def compose(self) -> ComposeResult:
        yield Header(name="VRAMGeist File Browser")
        with Horizontal():
            with Vertical(id="file_panel"):
                self.current_path_label = Static(f"📂 Current Directory: {self._format_path(self.start_dir)}")
                yield self.current_path_label
                yield Static("Use ↑↓ to navigate, Enter to show info, F2/Space to select, Q to cancel, . to toggle hidden files")
                self.file_browser = FileBrowserWidget(self.start_dir)
                yield self.file_browser
            # Metadata panel - initially hidden
            with Vertical(id="metadata_panel", classes="hidden"):
                yield Static("File Information", id="metadata_header")
                self.metadata_content = Static("", id="metadata_content")
                yield self.metadata_content
        yield Footer()
    
    def _format_path(self, path: Path) -> str:
        """Format path for display, using ~ for home directory."""
        try:
            return str(path).replace(str(Path.home()), "~")
        except:
            return str(path)
    
    def _update_path_display(self) -> None:
        """Update the current path display."""
        self.current_path_label.update(f"📂 Current Directory: {self._format_path(self.file_browser.current_dir)}")
    
    def action_cursor_up(self) -> None:
        """Move cursor up."""
        self.file_browser.action_cursor_up()
    
    def action_cursor_down(self) -> None:
        """Move cursor down."""
        self.file_browser.action_cursor_down()
    
    def action_page_up(self) -> None:
        """Move up by 10 items."""
        for _ in range(10):
            self.file_browser.action_cursor_up()
    
    def action_page_down(self) -> None:
        """Move down by 10 items."""
        for _ in range(10):
            self.file_browser.action_cursor_down()
    
    def action_go_home(self) -> None:
        """Navigate to home directory."""
        self._hide_metadata_panel()
        self.file_browser.navigate_to(Path.home())
        self._update_path_display()
    
    def action_go_up(self) -> None:
        """Navigate up one directory level."""
        self._hide_metadata_panel()
        parent = self.file_browser.current_dir.parent
        if parent != self.file_browser.current_dir:  # Not at root
            self.file_browser.navigate_to(parent)
            self._update_path_display()
    
    def action_select_or_enter(self) -> None:
        """Handle Enter key - either navigate into directory or show file info."""
        item = self.file_browser.get_selected_item()
        if not item:
            return
        
        if item.is_parent:
            # Navigate to parent directory
            self._hide_metadata_panel()
            self.file_browser.navigate_to(item.path)
            self._update_path_display()
        elif item.is_dir:
            # Navigate into directory
            self._hide_metadata_panel()
            self.file_browser.navigate_to(item.path)
            self._update_path_display()
        else:
            # For files, show metadata panel
            self._show_file_metadata(item.path)
    
    def action_select_current(self) -> None:
        """Select current item and exit."""
        item = self.file_browser.get_selected_item()
        if not item:
            return
        
        # Check if selection is allowed based on type
        if item.is_dir and not self.select_dirs:
            return
        if not item.is_dir and not self.select_files:
            return
        if item.is_parent:
            return  # Can't select parent directory entry
        
        self.selected_path = item.path
        self.dismiss(self.selected_path)
    
    def action_toggle_hidden(self) -> None:
        """Toggle showing hidden files."""
        self.file_browser.toggle_hidden()
    
    def action_cancel_or_hide(self) -> None:
        """Cancel and exit, or hide metadata panel if showing."""
        if self.show_metadata:
            self._hide_metadata_panel()
        else:
            self.cancelled = True
            self.dismiss(None)
    
    def _show_file_metadata(self, file_path: Path) -> None:
        """Show metadata panel for the selected file."""
        self.metadata_file = file_path
        self.show_metadata = True
        
        # Generate metadata content
        metadata_text = self._generate_file_metadata(file_path)
        self.metadata_content.update(metadata_text)
        
        # Show the metadata panel
        metadata_panel = self.query_one("#metadata_panel")
        metadata_panel.remove_class("hidden")
    
    def _hide_metadata_panel(self) -> None:
        """Hide the metadata panel."""
        self.show_metadata = False
        self.metadata_file = None
        metadata_panel = self.query_one("#metadata_panel")
        metadata_panel.add_class("hidden")
    
    def _generate_file_metadata(self, file_path: Path) -> str:
        """Generate metadata text for a file."""
        try:
            stat = file_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            
            metadata_lines = [
                f"📄 {file_path.name}",
                "",
                f"Size: {stat.st_size:,} bytes ({size_mb:.2f} MB)",
                f"Modified: {stat.st_mtime}",
                f"Type: {file_path.suffix or 'No extension'}",
                ""
            ]
            
            # Special handling for GGUF files
            if file_path.suffix.lower() == '.gguf':
                metadata_lines.extend([
                    "🤖 GGUF Model File",
                    "Press 'P' to analyze with VRAMGeist",
                    ""
                ])
                
                # Try to get basic GGUF info
                try:
                    metadata_lines.append("GGUF Analysis:")
                    metadata_lines.append("(Analysis would go here)")
                except Exception as e:
                    metadata_lines.append(f"GGUF analysis failed: {e}")
            
            return "\n".join(metadata_lines)
            
        except Exception as e:
            return f"Error reading file metadata: {e}"


def browse_files(start_dir: Path = None, select_files: bool = True, select_dirs: bool = True) -> Optional[Path]:
    """
    Open a Textual-based file browser and return the selected path.
    
    Args:
        start_dir: Starting directory (defaults to current working directory)
        select_files: Whether files can be selected
        select_dirs: Whether directories can be selected
    
    Returns:
        Selected Path object, or None if cancelled
    """
    class FileBrowserApp(App):
        def __init__(self):
            super().__init__()
            self.result: Optional[Path] = None
            
        def on_mount(self) -> None:
            screen = FileBrowserScreen(start_dir, select_files, select_dirs)
            
            def on_screen_result(result):
                self.result = result
                self.exit(result)
                
            self.push_screen(screen, callback=on_screen_result)
    
    app = FileBrowserApp()
    app.run()
    return app.result
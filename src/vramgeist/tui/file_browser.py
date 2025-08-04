from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static
from textual.containers import Vertical, Horizontal
from textual import events


class FileBrowserApp(App):
    """Simple file browser with direct key handling."""
    
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
    
    #file_header, #metadata_header {
        text-style: bold;
        color: $primary;
        height: 1;
    }
    """
    
    def __init__(self, start_dir: Path = None, select_files: bool = True, select_dirs: bool = True):
        super().__init__()
        self.current_dir = (start_dir or Path.cwd()).resolve()
        self.select_files = select_files
        self.select_dirs = select_dirs
        self.show_hidden = False
        self.selected_index = 0
        self.entries = []
        self.viewport_top = 0  # First visible item index
        self.viewport_height = 20  # Number of visible items (will be adjusted based on screen)
        self.show_metadata = False
        self.metadata_content = ""
        
    def compose(self) -> ComposeResult:
        yield Header(name="VRAMGeist File Browser")
        yield Static("↑↓/jk: navigate | PgUp/PgDn: scroll | Enter: select/navigate | Space/F2: process | ./h: toggle hidden | Esc: hide panel/quit")
        self.path_display = Static("")
        yield self.path_display
        
        with Horizontal():
            # Left panel: File browser
            with Vertical(id="file_panel"):
                yield Static("[bold]Files[/bold]", id="file_header")
                self.file_display = Static("")
                yield self.file_display
            
            # Right panel: Metadata/processing results (initially hidden)
            with Vertical(id="metadata_panel", classes="hidden"):
                yield Static("[bold]Analysis Results[/bold]", id="metadata_header")
                self.metadata_display = Static("")
                yield self.metadata_display
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the file browser."""
        self.refresh_display()
    
    def update_viewport(self) -> None:
        """Update viewport to ensure selected item is visible."""
        if not self.entries:
            return
            
        # If selection is above viewport, scroll up
        if self.selected_index < self.viewport_top:
            self.viewport_top = self.selected_index
        
        # If selection is below viewport, scroll down
        elif self.selected_index >= self.viewport_top + self.viewport_height:
            self.viewport_top = self.selected_index - self.viewport_height + 1
        
        # Ensure viewport doesn't go below 0 or beyond entries
        self.viewport_top = max(0, min(self.viewport_top, max(0, len(self.entries) - self.viewport_height)))
    
    def refresh_display(self) -> None:
        """Refresh the entire display."""
        # Update path
        self.path_display.update(f"📂 {self.current_dir}")
        
        # Get directory entries
        self.entries = []
        
        try:
            # Add parent directory if not at root
            if self.current_dir.parent != self.current_dir:
                self.entries.append({
                    'path': self.current_dir.parent,
                    'name': '..',
                    'is_dir': True,
                    'is_parent': True,
                    'display': '↩ ..'
                })
            
            # Get all items in directory
            items = []
            all_items = list(self.current_dir.iterdir())
            for item in all_items:
                # Skip hidden files if not showing them
                if not self.show_hidden and item.name.startswith('.'):
                    continue
                items.append(item)
            
            # Sort: directories first, then files
            items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
            
            # Add to entries
            for item in items:
                if item.is_dir():
                    display = f"📁 {item.name}/"
                else:
                    display = f"📄 {item.name}"
                
                self.entries.append({
                    'path': item,
                    'name': item.name,
                    'is_dir': item.is_dir(),
                    'is_parent': False,
                    'display': display
                })
                
        except PermissionError:
            self.entries.append({
                'path': None,
                'name': 'Permission denied',
                'is_dir': False,
                'is_parent': False,
                'display': '❌ Permission denied'
            })
        
        # Ensure selected index is valid
        if self.selected_index >= len(self.entries):
            self.selected_index = max(0, len(self.entries) - 1)
        elif self.selected_index < 0:
            self.selected_index = 0
        
        # Reset viewport when refreshing directory
        self.viewport_top = 0
        self.update_viewport()
        self.update_file_list()
    
    def update_file_list(self) -> None:
        """Update the file list display with viewport scrolling."""
        lines = []
        
        if not self.entries:
            lines = ["<empty directory>"]
        else:
            # Get visible entries within viewport
            viewport_end = min(self.viewport_top + self.viewport_height, len(self.entries))
            visible_entries = self.entries[self.viewport_top:viewport_end]
            
            for i, entry in enumerate(visible_entries):
                actual_index = self.viewport_top + i
                if actual_index == self.selected_index:
                    lines.append(f"[reverse]{entry['display']}[/reverse]")
                else:
                    lines.append(entry['display'])
            
            # Add scroll indicators if needed
            if self.viewport_top > 0:
                lines.insert(0, "[dim]↑ More items above...[/dim]")
            if viewport_end < len(self.entries):
                lines.append("[dim]↓ More items below...[/dim]")
        
        # Add debug info
        total_items = len(list(self.current_dir.iterdir())) if self.current_dir.exists() else 0
        visible_items = len(self.entries) - (1 if self.entries and self.entries[0]['is_parent'] else 0)
        scroll_info = f"[{self.selected_index + 1}/{len(self.entries)}]" if self.entries else ""
        debug_line = f"[dim]Showing {visible_items}/{total_items} items {scroll_info} (hidden: {'on' if self.show_hidden else 'off'})[/dim]"
        lines.append(debug_line)
            
        self.file_display.update("\n".join(lines))
    
    def show_metadata_panel(self) -> None:
        """Show the metadata panel."""
        self.show_metadata = True
        metadata_panel = self.query_one("#metadata_panel")
        metadata_panel.remove_class("hidden")
    
    def hide_metadata_panel(self) -> None:
        """Hide the metadata panel."""
        self.show_metadata = False
        metadata_panel = self.query_one("#metadata_panel")
        metadata_panel.add_class("hidden")
    
    def process_gguf_file(self, file_path: Path) -> None:
        """Process a single GGUF file and display results."""
        try:
            # Import VRAMGeist processing functions
            from ..ui import analyze_gguf_file
            
            self.metadata_display.update("[bold yellow]Processing...[/bold yellow]")
            self.show_metadata_panel()
            
            # Analyze the file (function only takes path argument)
            result = analyze_gguf_file(str(file_path))
            
            # Format the results for display
            lines = []
            lines.append(f"[bold cyan]📄 {file_path.name}[/bold cyan]")
            lines.append("")
            
            # Hardware info
            if 'gpu_vram_mb' in result or 'total_ram_gb' in result:
                lines.append("[bold]Hardware Detected:[/bold]")
                if 'gpu_vram_mb' in result:
                    lines.append(f"  GPU VRAM: {result['gpu_vram_mb']} MB")
                if 'total_ram_gb' in result:
                    lines.append(f"  System RAM: {result['total_ram_gb']} GB")
                lines.append("")
            
            # Model info
            if 'model_size_mb' in result or 'layer_count' in result:
                lines.append("[bold]Model Information:[/bold]")
                if 'model_size_mb' in result:
                    size_mb = result['model_size_mb']
                    size_str = f"{size_mb:.1f} MB" if isinstance(size_mb, (int, float)) else str(size_mb)
                    lines.append(f"  Size: {size_str}")
                if 'layer_count' in result:
                    lines.append(f"  Layers: {result['layer_count']}")
                lines.append("")
            
            # Analysis results
            if 'analysis_results' in result:
                analysis = result['analysis_results']
                lines.append("[bold]Optimal Configuration:[/bold]")
                
                if 'best_gpu_layers' in analysis:
                    lines.append(f"  GPU Layers: {analysis['best_gpu_layers']}")
                
                if 'max_context' in analysis:
                    max_ctx = analysis['max_context']
                    ctx_str = f"{max_ctx:,}" if isinstance(max_ctx, (int, float)) else str(max_ctx)
                    lines.append(f"  Max Context: {ctx_str}")
                
                if 'vram_usage_mb' in analysis:
                    vram = analysis['vram_usage_mb']
                    vram_str = f"{vram:.1f} MB" if isinstance(vram, (int, float)) else str(vram)
                    lines.append(f"  VRAM Usage: {vram_str}")
                
                if 'ram_usage_mb' in analysis:
                    ram = analysis['ram_usage_mb']
                    ram_str = f"{ram:.1f} MB" if isinstance(ram, (int, float)) else str(ram)
                    lines.append(f"  RAM Usage: {ram_str}")
                
                lines.append("")
                
                # Show additional configurations if available
                if 'configurations' in analysis:
                    configs = analysis['configurations'][:5]  # Show top 5
                    if configs:
                        lines.append("[bold]Alternative Configurations:[/bold]")
                        for config in configs:
                            gpu_layers = config.get('gpu_layers', 'N/A')
                            max_ctx = config.get('max_context', 'N/A')
                            vram_usage = config.get('vram_usage', 'N/A')
                            
                            ctx_str = f"{max_ctx:,}" if isinstance(max_ctx, (int, float)) else str(max_ctx)
                            vram_str = f"{vram_usage:.0f}MB" if isinstance(vram_usage, (int, float)) else str(vram_usage)
                            
                            lines.append(f"  Layers: {gpu_layers}, Context: {ctx_str}, VRAM: {vram_str}")
                        lines.append("")
                
            # Warnings
            if 'warnings' in result and result['warnings']:
                lines.append("[bold yellow]Warnings:[/bold yellow]")
                for warning in result['warnings']:
                    lines.append(f"  ⚠ {warning}")
                lines.append("")
            
            self.metadata_content = "\n".join(lines)
            self.metadata_display.update(self.metadata_content)
            
        except Exception as e:
            error_msg = f"[bold red]Error processing {file_path.name}:[/bold red]\n{str(e)}"
            self.metadata_display.update(error_msg)
            self.show_metadata_panel()
    
    def process_directory(self, dir_path: Path) -> None:
        """Process all GGUF files in a directory and display summary."""
        try:
            # Find all GGUF files
            gguf_files = list(dir_path.glob("*.gguf"))
            
            if not gguf_files:
                self.metadata_display.update(f"[yellow]No GGUF files found in {dir_path.name}[/yellow]")
                self.show_metadata_panel()
                return
            
            self.metadata_display.update(f"[bold yellow]Processing {len(gguf_files)} GGUF files...[/bold yellow]")
            self.show_metadata_panel()
            
            # Import processing functions
            from ..ui import analyze_gguf_file
            
            lines = []
            lines.append(f"[bold cyan]📁 {dir_path.name}[/bold cyan]")
            lines.append(f"Found {len(gguf_files)} GGUF file(s)")
            lines.append("")
            
            # Process each file
            for gguf_file in sorted(gguf_files):
                try:
                    result = analyze_gguf_file(str(gguf_file))
                    
                    size_mb = result.get('model_size_mb', 0)
                    size_str = f"{size_mb:.1f}MB" if isinstance(size_mb, (int, float)) else str(size_mb)
                    
                    lines.append(f"[bold]📄 {gguf_file.name}[/bold]")
                    lines.append(f"  Size: {size_str}")
                    
                    if 'analysis_results' in result:
                        analysis = result['analysis_results']
                        gpu_layers = analysis.get('best_gpu_layers', 'N/A')
                        max_context = analysis.get('max_context', 'N/A')
                        vram_usage = analysis.get('vram_usage_mb', 0)
                        
                        ctx_str = f"{max_context:,}" if isinstance(max_context, (int, float)) else str(max_context)
                        vram_str = f"{vram_usage:.0f}MB" if isinstance(vram_usage, (int, float)) else str(vram_usage)
                        
                        lines.append(f"  Optimal: {gpu_layers} layers, {ctx_str} context, {vram_str} VRAM")
                    else:
                        lines.append("  Analysis failed")
                    
                    lines.append("")
                    
                except Exception as e:
                    lines.append(f"[bold]📄 {gguf_file.name}[/bold]")
                    lines.append(f"  [red]Error: {str(e)}[/red]")
                    lines.append("")
            
            self.metadata_content = "\n".join(lines)
            self.metadata_display.update(self.metadata_content)
            
        except Exception as e:
            error_msg = f"[bold red]Error processing directory:[/bold red]\n{str(e)}"
            self.metadata_display.update(error_msg)
            self.show_metadata_panel()
    
    async def on_key(self, event: events.Key) -> None:
        """Handle key presses directly."""
        key = event.key
        
        # Debug: show what key was pressed in the path display temporarily
        self.path_display.update(f"📂 {self.current_dir} [Key pressed: '{key}']")
        
        # Navigation
        if key in ("up", "k"):
            if self.selected_index > 0:
                self.selected_index -= 1
                self.update_viewport()
                self.update_file_list()
        
        elif key in ("down", "j"):
            if self.selected_index < len(self.entries) - 1:
                self.selected_index += 1
                self.update_viewport()
                self.update_file_list()
        
        elif key == "pageup":
            # Move up by viewport height
            self.selected_index = max(0, self.selected_index - self.viewport_height)
            self.update_viewport()
            self.update_file_list()
        
        elif key == "pagedown":
            # Move down by viewport height
            self.selected_index = min(len(self.entries) - 1, self.selected_index + self.viewport_height)
            self.update_viewport()
            self.update_file_list()
        
        elif key == "home":
            self.current_dir = Path.home()
            self.selected_index = 0
            self.refresh_display()
        
        elif key in ("left", "backspace"):
            parent = self.current_dir.parent
            if parent != self.current_dir:
                self.current_dir = parent
                self.selected_index = 0
                self.refresh_display()
        
        # Toggle hidden files - try multiple key variations
        elif key in ("period", ".", "full_stop", "h"):
            self.show_hidden = not self.show_hidden
            # Reset selection to prevent index out of bounds
            old_selected = self.selected_index
            self.selected_index = 0
            self.refresh_display()
            # Show feedback with count
            try:
                hidden_count = sum(1 for item in self.current_dir.iterdir() if item.name.startswith('.'))
                status = f"on ({hidden_count} hidden)" if self.show_hidden else "off"
                self.path_display.update(f"📂 {self.current_dir} [Hidden files: {status}]")
            except:
                status = "on" if self.show_hidden else "off"
                self.path_display.update(f"📂 {self.current_dir} [Hidden files: {status}]")
        
        # Enter - select/navigate (traditional file browser behavior)
        elif key == "enter":
            if not self.entries or self.selected_index >= len(self.entries):
                return
                
            entry = self.entries[self.selected_index]
            if entry['path'] is None:  # Permission denied
                return
            
            if entry['is_parent']:
                # Navigate to parent directory
                self.current_dir = entry['path'].resolve()
                self.selected_index = 0
                self.hide_metadata_panel()
                self.refresh_display()
            elif entry['is_dir']:
                # Navigate into directory
                self.current_dir = entry['path'].resolve()
                self.selected_index = 0
                self.hide_metadata_panel()
                self.refresh_display()
            else:
                # Select file
                if self.select_files:
                    self.exit(entry['path'])
        
        # Space/F2 - process GGUF files/directories
        elif key in ("space", "f2"):
            if not self.entries or self.selected_index >= len(self.entries):
                return
                
            entry = self.entries[self.selected_index]
            if entry['path'] is None or entry['is_parent']:
                return
            
            if entry['is_dir']:
                # Check if directory contains GGUF files for processing
                dir_path = entry['path']
                gguf_files = list(dir_path.glob("*.gguf"))
                
                if gguf_files:
                    # Process directory with GGUF files
                    self.process_directory(dir_path)
                else:
                    # No GGUF files, treat as regular directory selection
                    if self.select_dirs:
                        self.exit(dir_path)
            else:
                # Check if it's a GGUF file for processing
                file_path = entry['path']
                if file_path.suffix.lower() == '.gguf':
                    # Process GGUF file
                    self.process_gguf_file(file_path)
                else:
                    # Regular file selection
                    if self.select_files:
                        self.exit(file_path)
        
        # Quit or hide metadata panel
        elif key in ("q", "escape"):
            if self.show_metadata:
                # Hide metadata panel first
                self.hide_metadata_panel()
            else:
                # Exit application
                self.exit(None)


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
    app = FileBrowserApp(start_dir, select_files, select_dirs)
    return app.run()
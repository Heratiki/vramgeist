from __future__ import annotations

import os
import json
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
    
    def __init__(self, start_dir: Path = None, select_files: bool = True, select_dirs: bool = True, enable_validation: bool = True):
        super().__init__()
        self.current_dir = (start_dir or Path.cwd()).resolve()
        self.select_files = select_files
        self.select_dirs = select_dirs
        self.selected_index = 0
        self.entries = []
        self.viewport_top = 0  # First visible item index
        self.viewport_height = 20  # Number of visible items (will be adjusted based on screen)
        self.show_metadata = False
        self.metadata_content = ""
        
        # Load persistent settings
        self.settings_file = Path.home() / ".vramgeist_settings.json"
        self.show_hidden = self.load_settings().get("show_hidden", False)
        
        # Validation settings
        self.enable_validation = enable_validation
        self.validate_settings = False
        self.llama_bin = None
        self.validation_timeout = 30.0
        
        # Auto-load validation settings if available
        if enable_validation:
            try:
                from ..config_persist import get_llama_bin_path, get_validation_timeout, should_enable_validation_by_default
                self.llama_bin = get_llama_bin_path()
                self.validation_timeout = get_validation_timeout()
                self.validate_settings = should_enable_validation_by_default()
            except ImportError:
                pass
        
    def compose(self) -> ComposeResult:
        yield Header(name="VRAMGeist File Browser")
        yield Static("↑↓/jk: navigate | PgUp/PgDn: scroll | Enter: info/navigate | Space/F2: GGUF analysis | v: toggle validation | ./h: toggle hidden | Esc: hide panel/quit")
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
    
    def load_settings(self) -> dict:
        """Load persistent settings from file."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def save_settings(self) -> None:
        """Save current settings to file."""
        try:
            settings = {
                "show_hidden": self.show_hidden
            }
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f)
        except Exception:
            pass
    
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
    
    def show_file_metadata(self, file_path: Path) -> None:
        """Show file metadata and basic info (not VRAM processing)."""
        try:
            self.show_metadata_panel()
            
            lines = []
            lines.append(f"[bold cyan]📄 {file_path.name}[/bold cyan]")
            lines.append("")
            
            # Basic file info
            try:
                stat = file_path.stat()
                size_mb = stat.st_size / (1024 * 1024)
                lines.append("[bold]File Information:[/bold]")
                lines.append(f"  Size: {stat.st_size:,} bytes ({size_mb:.2f} MB)")
                lines.append(f"  Type: {file_path.suffix or 'No extension'}")
                lines.append("")
            except Exception as e:
                lines.append(f"  [red]Error reading file info: {e}[/red]")
                lines.append("")
            
            # GGUF-specific info
            if file_path.suffix.lower() == '.gguf':
                lines.append("[bold yellow]🤖 GGUF Model File[/bold yellow]")
                lines.append("")
                lines.append("[dim]Press [bold]Space[/bold] or [bold]F2[/bold] to run VRAM analysis[/dim]")
                lines.append("")
                
                # Try to get basic GGUF metadata without full processing
                try:
                    from ..gguf import parse_gguf_metadata
                    metadata = parse_gguf_metadata(str(file_path))
                    
                    if metadata:
                        lines.append("[bold]GGUF Metadata:[/bold]")
                        if 'model_name' in metadata:
                            lines.append(f"  Model: {metadata['model_name']}")
                        if 'layer_count' in metadata:
                            lines.append(f"  Layers: {metadata['layer_count']}")
                        if 'context_length' in metadata:
                            lines.append(f"  Context Length: {metadata['context_length']:,}")
                        lines.append("")
                        
                except Exception:
                    lines.append("[dim]GGUF metadata unavailable[/dim]")
                    lines.append("")
            else:
                lines.append("[dim]Not a GGUF model file[/dim]")
                lines.append("")
            
            self.metadata_content = "\n".join(lines)
            self.metadata_display.update(self.metadata_content)
            
        except Exception as e:
            error_msg = f"[bold red]Error reading file metadata:[/bold red]\n{str(e)}"
            self.metadata_display.update(error_msg)
            self.show_metadata_panel()
    
    def process_gguf_file(self, file_path: Path) -> None:
        """Process a single GGUF file with full VRAM analysis."""
        try:
            # Import VRAMGeist processing functions
            from .. import ui as ui_module
            from ..config import DEFAULT_CONFIG
            
            # Show initial analysis message
            validation_msg = ""
            if self.validate_settings and self.llama_bin:
                validation_msg = "\n[dim]⚠️  Validation enabled - this may take longer...[/dim]"
            
            self.metadata_display.update(f"[bold yellow]🔄 Running VRAM Analysis...{validation_msg}[/bold yellow]\n\n[dim]This may take a moment...[/dim]")
            self.show_metadata_panel()
            
            # Analyze the file with validation settings
            result = ui_module.analyze_gguf_file_with_config(
                str(file_path),
                config=DEFAULT_CONFIG,
                vram_override=None,
                ram_override=None,
                force_detect=False,
                optimize_for="balanced",
                gpu_bandwidth_gbps=None,
                measure_bandwidth=False,
                measure_tps=False,
                llama_bin=self.llama_bin if self.validate_settings else None,
                bench_contexts=None,
                debug=False,
                validate_settings=self.validate_settings,
                validation_timeout=self.validation_timeout,
            )
            
            # Format the comprehensive results
            lines = []
            lines.append(f"[bold cyan]🤖 {file_path.name}[/bold cyan]")
            lines.append("")
            
            # Hardware Detection Results
            lines.append("[bold green]💻 Hardware Detected:[/bold green]")
            system_info = result.get('system', {})
            available_vram = system_info.get('vram_available_mb', 'Unknown')
            total_ram_mb = system_info.get('ram_total_mb', 'Unknown')
            available_ram = system_info.get('ram_available_mb', 'Unknown')
            
            # Convert total RAM from MB to GB for display
            if isinstance(total_ram_mb, (int, float)):
                total_ram = total_ram_mb / 1024
            else:
                total_ram = total_ram_mb
            
            if isinstance(available_vram, (int, float)):
                lines.append(f"  GPU VRAM: {available_vram:,} MB")
            else:
                lines.append(f"  GPU VRAM: {available_vram}")
                
            if isinstance(total_ram, (int, float)):
                lines.append(f"  System RAM: {total_ram:.1f} GB")
            else:
                lines.append(f"  System RAM: {total_ram}")
                
            if isinstance(available_ram, (int, float)):
                lines.append(f"  Available RAM: {available_ram:,} MB")
            lines.append("")
            
            # Model Analysis Results
            lines.append("[bold blue]📊 Model Analysis:[/bold blue]")
            model_info = result.get('model', {})
            model_size = model_info.get('size_mb', 'Unknown')
            layer_count = model_info.get('layers', 'Unknown')
            
            if isinstance(model_size, (int, float)):
                lines.append(f"  File Size: {model_size:.1f} MB ({model_size/1024:.2f} GB)")
            else:
                lines.append(f"  File Size: {model_size}")
                
            lines.append(f"  Layer Count: {layer_count}")
            lines.append("")
            
            # Optimal Configuration
            lines.append("[bold magenta]🎯 Recommended Configuration:[/bold magenta]")
            recommendation = result.get('recommendation', {})
            best_gpu_layers = recommendation.get('gpu_layers', 'N/A')
            best_context = recommendation.get('max_context', 'N/A')
            best_vram_usage = recommendation.get('expected_vram_mb', 'N/A')
            best_ram_usage = recommendation.get('expected_ram_mb', 'N/A')
            
            lines.append(f"  🔧 GPU Layers: [bold yellow]{best_gpu_layers}[/bold yellow]")
            
            # Safely format context
            try:
                if best_context != 'N/A' and best_context is not None:
                    context_val = int(float(best_context))
                    lines.append(f"  📝 Max Context: [bold yellow]{context_val:,}[/bold yellow] tokens")
                else:
                    lines.append(f"  📝 Max Context: {best_context}")
            except (ValueError, TypeError):
                lines.append(f"  📝 Max Context: {best_context}")
                
            # Safely format VRAM usage
            try:
                if best_vram_usage != 'N/A' and best_vram_usage is not None:
                    vram_val = float(best_vram_usage)
                    lines.append(f"  🎮 VRAM Usage: [bold yellow]{vram_val:.0f} MB[/bold yellow]")
                else:
                    lines.append(f"  🎮 VRAM Usage: {best_vram_usage}")
            except (ValueError, TypeError):
                lines.append(f"  🎮 VRAM Usage: {best_vram_usage}")
                
            # Safely format RAM usage
            try:
                if best_ram_usage != 'N/A' and best_ram_usage is not None:
                    ram_val = float(best_ram_usage)
                    lines.append(f"  💾 RAM Usage: [bold yellow]{ram_val:.0f} MB[/bold yellow]")
                else:
                    lines.append(f"  💾 RAM Usage: {best_ram_usage}")
            except (ValueError, TypeError):
                lines.append(f"  💾 RAM Usage: {best_ram_usage}")
            lines.append("")
            
            # Alternative Configurations Table
            configurations = result.get('analysis', [])
            if configurations:
                lines.append("[bold cyan]⚙️ Alternative Configurations:[/bold cyan]")
                lines.append("[dim]GPU Layers | Max Context | VRAM Used | RAM Used[/dim]")
                lines.append("[dim]-----------|-------------|-----------|----------[/dim]")
                
                for config in configurations[:6]:  # Show top 6
                    gpu_layers = config.get('gpu_layers', 'N/A')
                    max_context = config.get('max_context', 'N/A')
                    vram_usage = config.get('vram_usage_mb', 'N/A')
                    ram_usage = config.get('ram_usage_mb', 'N/A')
                    
                    # Safely format numbers
                    try:
                        ctx_str = f"{int(max_context):,}" if max_context != 'N/A' and max_context is not None else str(max_context)
                    except (ValueError, TypeError):
                        ctx_str = str(max_context)
                    
                    try:
                        vram_str = f"{float(vram_usage):.0f}MB" if vram_usage != 'N/A' and vram_usage is not None else str(vram_usage)
                    except (ValueError, TypeError):
                        vram_str = str(vram_usage)
                    
                    try:
                        ram_str = f"{float(ram_usage):.0f}MB" if ram_usage != 'N/A' and ram_usage is not None else str(ram_usage)
                    except (ValueError, TypeError):
                        ram_str = str(ram_usage)
                    
                    # Format GPU layers safely
                    try:
                        gpu_str = f"{int(gpu_layers):>2}" if gpu_layers != 'N/A' and gpu_layers is not None else f"{str(gpu_layers):>2}"
                    except (ValueError, TypeError):
                        gpu_str = f"{str(gpu_layers):>2}"
                    
                    lines.append(f"    {gpu_str} | {ctx_str:>11} | {vram_str:>9} | {ram_str:>8}")
                lines.append("")
            
            # Performance Notes
            try:
                gpu_layers_int = int(best_gpu_layers) if isinstance(best_gpu_layers, (int, str)) else 0
                layer_count_int = int(layer_count) if isinstance(layer_count, (int, str)) else 0
                
                if gpu_layers_int == layer_count_int and layer_count_int > 0:
                    lines.append("[bold green]✅ Full GPU acceleration possible![/bold green]")
                elif gpu_layers_int > 0:
                    lines.append(f"[bold yellow]⚡ Hybrid GPU/CPU processing ({gpu_layers_int}/{layer_count_int} layers on GPU)[/bold yellow]")
                else:
                    lines.append("[bold red]🐌 CPU-only processing (limited VRAM)[/bold red]")
            except (ValueError, TypeError):
                lines.append("[dim]Performance analysis unavailable[/dim]")
            lines.append("")
            
            # Warnings and Recommendations
            warnings = result.get('warnings', [])
            if warnings:
                lines.append("[bold yellow]⚠️ Warnings & Recommendations:[/bold yellow]")
                for warning in warnings:
                    lines.append(f"  • {warning}")
                lines.append("")
            
            # Validation Results
            validation = result.get('validation')
            if validation:
                lines.append("[bold magenta]🔍 Validation Results:[/bold magenta]")
                if validation.get('validated'):
                    lines.append("[green]  🔒 PASSED: Settings validated with llama.cpp[/green]")
                    lines.append(f"  ✅ Model loaded successfully with {validation.get('tested_gpu_layers', '?')} GPU layers")
                    lines.append(f"  ✅ Context length {validation.get('tested_context', '?'):,} confirmed working")
                else:
                    reason = validation.get('reason', 'Unknown error')
                    lines.append(f"[red]  ⚠️  FAILED: {reason}[/red]")
                    
                    # Show debug information for troubleshooting
                    details = validation.get('details', {})
                    if details.get('debug_command'):
                        lines.append(f"[dim]  Debug - Command: {details['debug_command']}[/dim]")
                    if details.get('stderr_sample'):
                        stderr = details['stderr_sample'][:100] + "..." if len(details.get('stderr_sample', '')) > 100 else details.get('stderr_sample', '')
                        if stderr:
                            lines.append(f"[dim]  Debug - Error: {stderr}[/dim]")
                    
                    # Show fallback recommendations
                    recommendations = validation.get('recommendations', [])
                    for rec_text in recommendations:
                        if 'Fallback validated' in rec_text:
                            lines.append(f"[yellow]  {rec_text}[/yellow]")
                        else:
                            lines.append(f"[dim]  {rec_text}[/dim]")
                lines.append("")
            elif self.validate_settings:
                if self.llama_bin:
                    lines.append("[bold magenta]🔍 Validation:[/bold magenta]")
                    lines.append("[yellow]  ⚠️  Validation was requested but not performed[/yellow]")
                else:
                    lines.append("[bold magenta]🔍 Validation:[/bold magenta]")
                    lines.append("[dim]  ❌ Validation unavailable (no llama.cpp binary configured)[/dim]")
                lines.append("")
            
            # Add usage tip
            validation_tip = ""
            if validation and validation.get('validated'):
                validation_tip = " (✅ Validated with llama.cpp)"
            elif validation and not validation.get('validated'):
                validation_tip = " (⚠️ Validation failed - consider fallback values)"
            
            lines.append(f"[dim]💡 Tip: Use these values in your AI software's settings{validation_tip}[/dim]")
            
            self.metadata_content = "\n".join(lines)
            self.metadata_display.update(self.metadata_content)
            
        except Exception as e:
            error_msg = f"[bold red]❌ Analysis Failed:[/bold red]\n\n{str(e)}\n\n[dim]Check that the file is a valid GGUF model[/dim]"
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
        
        # Toggle validation: 'v'
        elif key == "v":
            if self.enable_validation:
                self.validate_settings = not self.validate_settings
                status = "ON" if self.validate_settings else "OFF"
                if self.llama_bin:
                    self.path_display.update(f"📂 {self.current_dir} [Validation: {status}]")
                else:
                    self.path_display.update(f"📂 {self.current_dir} [Validation: UNAVAILABLE - no llama.cpp path]")
            else:
                self.path_display.update(f"📂 {self.current_dir} [Validation: DISABLED]")
        
        # Toggle hidden files - try multiple key variations
        elif key in ("period", ".", "full_stop", "h"):
            self.show_hidden = not self.show_hidden
            # Save settings persistently
            self.save_settings()
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
        
        # Enter - show metadata/navigate (don't process)
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
                # Show file metadata (not process)
                self.show_file_metadata(entry['path'])
        
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
                    # Process GGUF file in metadata panel
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


def browse_files(start_dir: Path = None, select_files: bool = True, select_dirs: bool = True, enable_validation: bool = True) -> Optional[Path]:
    """
    Open a Textual-based file browser and return the selected path.
    
    Args:
        start_dir: Starting directory (defaults to current working directory)
        select_files: Whether files can be selected
        select_dirs: Whether directories can be selected
        enable_validation: Whether to enable validation features
    
    Returns:
        Selected Path object, or None if cancelled
    """
    app = FileBrowserApp(start_dir, select_files, select_dirs, enable_validation)
    return app.run()